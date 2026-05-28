import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import rag_10k
from rag_10k import (
    DEFAULT_MODEL,
    build_or_load_index,
    chunk_section,
    chunk_text,
    make_retrieval_tool,
    retrieve,
    split_sections,
)


FAKE_DIM = 16


def _fake_embed(texts, model_name):
    """
    Deterministic fake embedder used in tests. Seeds a numpy RNG from
    sha256(model_name + text) and draws an L2-normalized FAKE_DIM
    vector. Independent of FAKE_DIM so the helper is robust to digest
    length, and the model_name is woven in so cache-invalidation-on-
    model-change can be verified.
    """
    matrix = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)

    for i, text in enumerate(texts):
        digest = hashlib.sha256((model_name + "::" + text).encode()).digest()
        seed_int = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed_int)
        vec = rng.standard_normal(FAKE_DIM).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        matrix[i] = vec / (norm if norm > 0 else 1.0)

    return matrix


def _install_fake_embedder(monkeypatch, call_log=None):
    def wrapped(texts, model_name):
        if call_log is not None:
            call_log.append({"n": len(texts), "model": model_name})
        return _fake_embed(texts, model_name)

    monkeypatch.setattr(rag_10k, "_embed_texts", wrapped)


def _make_10k(sections):
    """
    Build a synthetic 10-K-like text with a fake table of contents
    followed by body sections of the requested lengths.
    """
    toc_lines = []
    body_parts = []

    for code, title, body in sections:
        toc_lines.append(f"Item {code}.\n\n{title}\n\n1\n")
        body_parts.append(f"Item {code}. {title}\n\n{body}\n\n")

    return "\n".join(toc_lines) + "\n\n" + "".join(body_parts)


def test_split_sections_finds_item_markers():
    text = _make_10k([
        ("1", "Business", "A" * 500),
        ("1A", "Risk Factors", "B" * 500),
        ("7", "MD&A", "C" * 500),
    ])

    sections = split_sections(text)

    codes = [s["section_code"] for s in sections]

    assert codes == ["1", "1A", "7"]


def test_split_sections_drops_short_toc_entries():
    text = _make_10k([
        ("1", "Business", "X" * 500),
        ("1A", "Risk Factors", "Y" * 500),
    ])

    sections = split_sections(text)

    assert len(sections) == 2
    assert all(len(s["text"]) >= 200 for s in sections)


def test_split_sections_handles_amzn_inline_title():
    body = "Z" * 500
    text = f"Item 1A.Risk Factors\n\n{body}\n\nItem 2. Properties\n\n{'Q' * 500}"

    sections = split_sections(text)

    assert sections[0]["section_code"] == "1A"
    assert sections[0]["section_title"] == "Risk Factors"


def test_split_sections_handles_amgn_caps_title():
    body = "Z" * 500
    text = f"Item 1.BUSINESS\n\n{body}\n\nItem 1A. Risk Factors\n\n{'Q' * 500}"

    sections = split_sections(text)

    assert sections[0]["section_code"] == "1"
    assert sections[0]["section_title"] == "BUSINESS"


def test_split_sections_case_insensitive_item_marker():
    body = "Z" * 500
    text = f"ITEM 1A. Risk Factors\n\n{body}\n\nitem 2. Properties\n\n{'Q' * 500}"

    sections = split_sections(text)

    codes = [s["section_code"] for s in sections]

    assert codes == ["1A", "2"]


def test_split_sections_empty_when_no_markers():
    assert split_sections("Just some prose, no Item headers here.") == []


def test_split_sections_handles_no_separator_then_title():
    """
    MCD/HON-style TOC: "Item 1ARisk Factors" - no period, no space
    between the item code and the title (title starts with a capital).
    """
    body = "Z" * 500
    text = f"Item 1ARisk Factors\n\n{body}\n\nItem 2Properties\n\n{'Q' * 500}"

    sections = split_sections(text)

    assert [s["section_code"] for s in sections] == ["1A", "2"]
    assert sections[0]["section_title"] == "Risk Factors"
    assert sections[1]["section_title"] == "Properties"


def test_split_sections_handles_two_digit_no_separator():
    """
    HON-style: "Item 10Directors" - two-digit code directly followed
    by capital title-start. Must parse code as 10, not as 1.
    """
    body = "Z" * 500
    text = f"Item 10Directors and Officers\n\n{body}\n\nItem 11Compensation\n\n{'Q' * 500}"

    sections = split_sections(text)

    assert [s["section_code"] for s in sections] == ["10", "11"]
    assert sections[0]["section_title"] == "Directors and Officers"


def test_split_sections_handles_bracket_separator():
    """
    "Item 6[Reserved]" - bracket as the separator instead of period.
    """
    body = "Z" * 500
    text = f"Item 6[Reserved]\n\n{body}\n\nItem 7. MD&A\n\n{'Q' * 500}"

    sections = split_sections(text)

    assert sections[0]["section_code"] == "6"
    assert sections[1]["section_code"] == "7"


def test_split_sections_does_not_consume_letter_into_code_when_title_is_lowercase():
    """
    'Item 1Business' must parse as code 1 + title 'Business', NOT as
    code 1B + title 'usiness'. The optional letter is only treated as
    part of the code when the following character is uppercase.
    """
    body = "Z" * 500
    text = f"Item 1Business\n\n{body}\n\nItem 2Properties\n\n{'Q' * 500}"

    sections = split_sections(text)

    assert sections[0]["section_code"] == "1"
    assert sections[0]["section_title"] == "Business"


def test_chunk_section_respects_size_and_overlap():
    section = {
        "section_code": "1A",
        "section_title": "Risk Factors",
        "text": "x" * 2500,
        "char_start": 100,
        "char_end": 2600,
    }

    chunks = chunk_section(section, chunk_size=1000, overlap=150)

    assert len(chunks) == 3
    assert len(chunks[0]["text"]) == 1000
    assert chunks[1]["char_start"] - chunks[0]["char_start"] == 850
    assert all(c["section_code"] == "1A" for c in chunks)
    assert chunks[-1]["char_end"] == 2600


def test_chunk_section_single_chunk_when_smaller_than_chunk_size():
    section = {
        "section_code": "1",
        "section_title": "Business",
        "text": "short body",
        "char_start": 0,
        "char_end": 10,
    }

    chunks = chunk_section(section, chunk_size=1000, overlap=150)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "short body"


def test_chunk_section_rejects_overlap_ge_chunk_size():
    section = {
        "section_code": "1",
        "section_title": "",
        "text": "x" * 500,
        "char_start": 0,
        "char_end": 500,
    }

    with pytest.raises(ValueError):
        chunk_section(section, chunk_size=200, overlap=200)


def test_chunk_text_assigns_monotonic_global_ids():
    text = _make_10k([
        ("1", "Business", "A" * 2500),
        ("1A", "Risk Factors", "B" * 2500),
    ])

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    ids = [c["chunk_id"] for c in chunks]

    assert ids == list(range(len(chunks)))


def test_chunk_text_carries_section_metadata():
    text = _make_10k([
        ("1", "Business", "A" * 2500),
        ("1A", "Risk Factors", "B" * 2500),
    ])

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    codes = sorted({c["section_code"] for c in chunks})

    assert codes == ["1", "1A"]
    assert any(c["section_title"] == "Risk Factors" for c in chunks)


def test_chunk_text_falls_back_to_whole_text_when_no_sections():
    """
    HON / MCD style: body uses bare 'RISK FACTORS' headers instead
    of 'Item 1A. Risk Factors'. split_sections returns no sections,
    so chunk_text must still emit chunks (over the whole text)
    with empty section_code so the index builds.
    """
    text = "RISK FACTORS\n\n" + ("Z" * 3000) + "\n\nPROPERTIES\n\n" + ("Y" * 1500)

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    assert len(chunks) > 0
    assert all(c["section_code"] == "" for c in chunks)


def test_chunk_text_does_not_use_fallback_when_sections_present():
    text = _make_10k([
        ("1A", "Risk Factors", "B" * 2500),
    ])

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    assert all(c["section_code"] == "1A" for c in chunks)


def test_chunk_text_falls_back_when_section_coverage_is_low():
    """
    MCD-style: a dense TOC at the end of the document matches the
    regex many times but only the last entry's body extends far
    enough to pass MIN_SECTION_CHARS. The bulk of real text sits
    above the TOC and has no Item markers, so chunking on that one
    surviving section throws away ~99% of the document. The
    coverage threshold catches this and triggers the fallback.
    """
    pre_body = "Z" * 30_000  # 30 KB of real unmarked content
    toc = "\n\n".join(
        f"Item {i}TocTitle Page {i}" for i in range(1, 16)
    )
    trailing_body = "Item 16TrailingSection\n\n" + ("Y" * 500)

    text = pre_body + "\n\n" + toc + "\n\n" + trailing_body

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    assert {c["section_code"] for c in chunks} == {""}
    assert len(chunks) > 20


def test_build_or_load_index_writes_chunks_and_embeddings(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([
        ("1A", "Risk Factors", "supply chain dependency on China " * 50),
    ])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    assert index["ticker"] == "AAPL"
    assert index["accession_number"] == "ACC-1"
    assert index["model_name"] == "fake"
    assert len(index["chunks"]) > 0
    assert index["embeddings"].shape == (len(index["chunks"]), FAKE_DIM)
    assert (tmp_path / "AAPL" / "ACC-1.chunks.json").exists()
    assert (tmp_path / "AAPL" / "ACC-1.embeddings.npz").exists()


def test_build_or_load_index_returns_cached_on_second_call(monkeypatch, tmp_path):
    call_log = []
    _install_fake_embedder(monkeypatch, call_log=call_log)

    text = _make_10k([("1A", "Risk Factors", "R" * 2500)])

    first = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )
    first_calls = len(call_log)

    second = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    assert len(call_log) == first_calls
    assert second["chunks"] == first["chunks"]
    np.testing.assert_array_equal(second["embeddings"], first["embeddings"])


def test_build_or_load_index_rebuilds_on_model_name_change(monkeypatch, tmp_path):
    call_log = []
    _install_fake_embedder(monkeypatch, call_log=call_log)

    text = _make_10k([("1A", "Risk Factors", "R" * 2500)])

    build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="model-A",
    )
    calls_after_first = len(call_log)

    build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="model-B",
    )

    assert len(call_log) > calls_after_first
    assert call_log[-1]["model"] == "model-B"


def test_build_or_load_index_no_cache_dir_skips_disk(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([("1A", "Risk Factors", "R" * 2500)])

    index = build_or_load_index("AAPL", "ACC-1", text, cache_dir=None)

    assert len(index["chunks"]) > 0
    assert list(tmp_path.iterdir()) == []


def test_build_or_load_index_raises_when_text_is_empty(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    with pytest.raises(ValueError):
        build_or_load_index(
            "AAPL", "ACC-1",
            "   \n  \n",
            cache_dir=tmp_path,
        )


def test_build_or_load_index_uses_fallback_for_unmarked_text(monkeypatch, tmp_path):
    """
    Plain prose with no Item markers (HON/MCD-style) must still
    produce a valid index via the whole-text fallback.
    """
    _install_fake_embedder(monkeypatch)

    text = "Some long company description without item headers. " * 50

    index = build_or_load_index(
        "HON", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    assert len(index["chunks"]) > 0
    assert all(c["section_code"] == "" for c in index["chunks"])


def test_retrieve_returns_top_k_sorted_by_similarity(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([
        ("1A", "Risk Factors", "alpha beta gamma " * 200),
        ("7", "MD&A", "delta epsilon zeta " * 200),
    ])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    hits = retrieve(index, "alpha beta gamma", k=3)

    assert len(hits) == 3
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_retrieve_section_filter_string(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([
        ("1A", "Risk Factors", "alpha beta gamma " * 200),
        ("7", "MD&A", "delta epsilon zeta " * 200),
    ])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    hits = retrieve(index, "query text", k=10, section="1A")

    assert hits
    assert all(h["section_code"] == "1A" for h in hits)


def test_retrieve_section_filter_list(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([
        ("1A", "Risk Factors", "alpha " * 200),
        ("7", "MD&A", "beta " * 200),
        ("7A", "Quant Risk", "gamma " * 200),
    ])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    hits = retrieve(index, "query", k=100, section=["1A", "7"])
    codes = {h["section_code"] for h in hits}

    assert codes <= {"1A", "7"}
    assert "1A" in codes and "7" in codes


def test_retrieve_section_filter_case_insensitive(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([("1A", "Risk Factors", "alpha " * 200)])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    hits = retrieve(index, "query", k=5, section="1a")

    assert hits
    assert all(h["section_code"] == "1A" for h in hits)


def test_retrieve_empty_when_section_has_no_chunks(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([("1A", "Risk Factors", "alpha " * 200)])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    assert retrieve(index, "q", k=5, section="99Z") == []


def test_retrieve_k_capped_to_corpus_size(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([("1A", "Risk Factors", "alpha " * 200)])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    hits = retrieve(index, "q", k=10_000)

    assert len(hits) == len(index["chunks"])


def test_make_retrieval_tool_callable_signature(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([
        ("1A", "Risk Factors", "alpha " * 200),
        ("7", "MD&A", "beta " * 200),
    ])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    tool = make_retrieval_tool(index)

    out = tool("any query", k=2, section="1A")

    assert isinstance(out, str)
    assert "chunk " in out
    assert "Item 1A" in out
    assert "sim=" in out


def test_make_retrieval_tool_reports_empty(monkeypatch, tmp_path):
    _install_fake_embedder(monkeypatch)

    text = _make_10k([("1A", "Risk Factors", "alpha " * 200)])

    index = build_or_load_index(
        "AAPL", "ACC-1", text,
        cache_dir=tmp_path, model_name="fake",
    )

    tool = make_retrieval_tool(index)

    out = tool("q", k=5, section="99Z")

    assert "no chunks" in out.lower()


def test_end_to_end_with_real_embedder(tmp_path):
    """
    Smoke test with the real sentence-transformers embedder. Skipped
    when the library is not installed, so unit-test runs stay offline.
    """
    pytest.importorskip("sentence_transformers")

    text = _make_10k([
        ("1A", "Risk Factors",
         "Our supply chain in China faces geopolitical tariff risk. " * 30),
        ("7", "MD&A",
         "Net sales increased due to higher iPhone unit volume. " * 30),
    ])

    index = build_or_load_index(
        "AAPL", "ACC-REAL", text,
        cache_dir=tmp_path, model_name=DEFAULT_MODEL,
    )

    hits = retrieve(index, "tariffs and China supply risk", k=2, section="1A")

    assert hits
    assert all(h["section_code"] == "1A" for h in hits)
    assert hits[0]["similarity"] > 0.3
