import json
import re
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
MIN_SECTION_CHARS = 200
MIN_SECTION_COVERAGE = 0.3

SECTION_RE = re.compile(
    r"^(?i:Item)\s+(\d+[A-Z]?)"
    r"(?:[.\[]\s*|(?=[A-Z])|\s*$)"
    r"([^\n]*)$",
    re.MULTILINE,
)


def split_sections(text):
    """
    Split a 10-K's plain text into sections keyed by Item code.

    Walk all "^Item N[A-Z]?." matches and slice the body between
    consecutive matches. Drop sections whose body is shorter than
    MIN_SECTION_CHARS so the table of contents at the top of the
    filing does not leak through as 17 empty sections.

    Returns a list of dicts:
        {
            "section_code": "1A",
            "section_title": "Risk Factors",
            "text": "...",
            "char_start": int,
            "char_end": int,
        }
    """
    matches = list(SECTION_RE.finditer(text))

    if not matches:
        return []

    sections = []

    for i, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        body = text[body_start:body_end].strip()

        if len(body) < MIN_SECTION_CHARS:
            continue

        sections.append({
            "section_code": match.group(1).upper(),
            "section_title": match.group(2).strip(),
            "text": body,
            "char_start": body_start,
            "char_end": body_end,
        })

    return sections


def chunk_section(
    section,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_CHUNK_OVERLAP,
):
    """
    Size-chunk one section's text into pieces of `chunk_size` chars
    with `overlap` chars of overlap between consecutive chunks.

    Each chunk inherits section_code and section_title from the input
    section; char_start / char_end are offsets in the source text.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    body = section["text"]
    base_offset = section["char_start"]

    chunks = []
    step = chunk_size - overlap
    cursor = 0

    while cursor < len(body):
        end = min(cursor + chunk_size, len(body))

        chunks.append({
            "section_code": section["section_code"],
            "section_title": section["section_title"],
            "text": body[cursor:end],
            "char_start": base_offset + cursor,
            "char_end": base_offset + end,
        })

        if end == len(body):
            break

        cursor += step

    return chunks


def chunk_text(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_CHUNK_OVERLAP,
):
    """
    Full chunking pipeline: split into sections, size-chunk each,
    return a flat list of chunk dicts with a global chunk_id.

    Fallback: some 10-K filings (e.g. HON, MCD) render their body
    section headers as plain titles like RISK FACTORS rather than
    Item-prefixed lines, so the only matches come from a dense TOC.
    The fallback triggers when:

        a. no sections survive MIN_SECTION_CHARS, OR
        b. surviving sections together cover less than
           MIN_SECTION_COVERAGE of the document length - which
           happens for MCD, where 18 TOC entries collapse to one
           tiny surviving section that is not real body content.

    In either case the whole document is chunked as one sectionless
    section so retrieval still works - just without the section-code
    filter for that filing.
    """
    sections = split_sections(text)

    coverage = (
        sum(len(s["text"]) for s in sections) / len(text)
        if text else 0.0
    )

    if not sections or coverage < MIN_SECTION_COVERAGE:
        body = text.strip()

        if body:
            sections = [{
                "section_code": "",
                "section_title": "",
                "text": body,
                "char_start": 0,
                "char_end": len(text),
            }]
        else:
            sections = []

    chunks = []

    for section in sections:
        for chunk in chunk_section(section, chunk_size, overlap):
            chunk["chunk_id"] = len(chunks)
            chunks.append(chunk)

    return chunks


_embedder_cache = {}


def _load_embedder(model_name):
    """
    Lazy SentenceTransformer loader, memoized per model name so a
    batch run does not pay the model-load cost on every call. Imports
    inside the function so tests that monkeypatch _embed_texts never
    need the dependency.
    """
    if model_name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer

        _embedder_cache[model_name] = SentenceTransformer(model_name)

    return _embedder_cache[model_name]


def _embed_texts(texts, model_name):
    """
    Embed a list of strings and return an (N, d) float32 numpy array
    with L2-normalized rows so cosine similarity is a single dot.
    """
    model = _load_embedder(model_name)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.astype(np.float32)


def _chunks_path(cache_dir, ticker, accession_number):
    return cache_dir / ticker.upper() / f"{accession_number}.chunks.json"


def _embeddings_path(cache_dir, ticker, accession_number):
    return cache_dir / ticker.upper() / f"{accession_number}.embeddings.npz"


def _read_cache(cache_dir, ticker, accession_number, model_name):
    """
    Return a fully hydrated index dict if a valid cached entry exists,
    otherwise None. A model_name mismatch invalidates the cache so
    different embedding spaces never get mixed.
    """
    chunks_path = _chunks_path(cache_dir, ticker, accession_number)
    embeddings_path = _embeddings_path(cache_dir, ticker, accession_number)

    if not (chunks_path.exists() and embeddings_path.exists()):
        return None

    npz = np.load(embeddings_path, allow_pickle=False)

    cached_model = str(npz["model_name"])

    if cached_model != model_name:
        return None

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return {
        "ticker": ticker.upper(),
        "accession_number": accession_number,
        "model_name": cached_model,
        "chunks": chunks,
        "embeddings": npz["embeddings"].astype(np.float32),
    }


def _atomic_write_bytes(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        f.write(data)

    tmp_path.replace(path)


def _atomic_write_text(path, content):
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    tmp_path.replace(path)


def _write_cache(cache_dir, ticker, accession_number, model_name, chunks, embeddings):
    ticker_dir = cache_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = _chunks_path(cache_dir, ticker, accession_number)
    embeddings_path = _embeddings_path(cache_dir, ticker, accession_number)

    _atomic_write_text(chunks_path, json.dumps(chunks))

    import io

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        embeddings=embeddings,
        model_name=np.array(model_name),
    )

    _atomic_write_bytes(embeddings_path, buffer.getvalue())


def build_or_load_index(
    ticker,
    accession_number,
    text,
    cache_dir=None,
    model_name=DEFAULT_MODEL,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_CHUNK_OVERLAP,
):
    """
    Build a retrieval index for one 10-K filing, or load it from cache.

    Steps:
        1. If cache_dir has a valid <ticker>/<accession>.* entry whose
           model_name matches, return it without touching the embedder.
        2. Otherwise chunk the text, embed all chunks, and (if cache_dir
           is set) persist chunks.json and embeddings.npz atomically.

    Returns an index dict consumed by retrieve().
    """
    ticker_upper = ticker.upper()

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cached = _read_cache(cache_dir, ticker_upper, accession_number, model_name)

        if cached is not None:
            return cached

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        raise ValueError(
            f"No chunks produced for {ticker_upper} {accession_number}. "
            "The text may be empty or contain no recognizable Item markers."
        )

    embeddings = _embed_texts([c["text"] for c in chunks], model_name)

    if cache_dir is not None:
        _write_cache(
            cache_dir, ticker_upper, accession_number,
            model_name, chunks, embeddings,
        )

    return {
        "ticker": ticker_upper,
        "accession_number": accession_number,
        "model_name": model_name,
        "chunks": chunks,
        "embeddings": embeddings,
    }


def _normalize_section_filter(section):
    """
    Accept None / str / iterable[str], return None or a set of
    upper-cased codes for membership testing.
    """
    if section is None:
        return None

    if isinstance(section, str):
        return {section.upper()}

    return {str(s).upper() for s in section}


def retrieve(index, query, k=5, section=None):
    """
    Return the top-k chunks most similar to `query`.

    Optional section filter (str or list of str, case-insensitive)
    restricts results to chunks whose section_code is in the set.
    Each returned chunk has a "similarity" float added.
    """
    section_set = _normalize_section_filter(section)

    chunks = index["chunks"]
    embeddings = index["embeddings"]
    model_name = index["model_name"]

    if section_set is not None:
        candidate_indices = [
            i for i, c in enumerate(chunks)
            if c["section_code"].upper() in section_set
        ]
    else:
        candidate_indices = list(range(len(chunks)))

    if not candidate_indices:
        return []

    query_vec = _embed_texts([query], model_name)[0]

    candidate_embeddings = embeddings[candidate_indices]
    similarities = candidate_embeddings @ query_vec

    k_effective = min(k, len(candidate_indices))

    if k_effective < len(candidate_indices):
        top_local = np.argpartition(-similarities, k_effective - 1)[:k_effective]
    else:
        top_local = np.arange(len(candidate_indices))

    top_local = top_local[np.argsort(-similarities[top_local])]

    results = []

    for local_idx in top_local:
        global_idx = candidate_indices[int(local_idx)]
        chunk = dict(chunks[global_idx])
        chunk["similarity"] = float(similarities[int(local_idx)])
        results.append(chunk)

    return results


def _format_hits(hits):
    if not hits:
        return "(no chunks matched)"

    blocks = []

    for hit in hits:
        title = hit["section_title"] or "(untitled)"
        header = (
            f"[chunk {hit['chunk_id']} | "
            f"Item {hit['section_code']}. {title} | "
            f"sim={hit['similarity']:.3f}]"
        )
        blocks.append(f"{header}\n{hit['text']}")

    return "\n\n".join(blocks)


def make_retrieval_tool(index):
    """
    Build a closure-bound retrieval callable suitable for LangGraph
    tool binding. The returned function has the signature

        retrieve_10k(query: str, k: int = 5, section: str | None = None) -> str

    and returns a formatted multi-chunk string with citations.
    """

    def retrieve_10k(query, k=5, section=None):
        """
        Return the top-k 10-K chunks most relevant to `query`. Pass
        `section` to restrict to a single 10-K item (e.g. "1A" for
        Risk Factors, "7" for MD&A). Each chunk is annotated with its
        chunk id, section, and similarity score.
        """
        hits = retrieve(index, query, k=k, section=section)
        return _format_hits(hits)

    return retrieve_10k
