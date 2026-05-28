import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from edgar_retrieval import (
    DOW_30,
    NY_TZ,
    build_primary_doc_url,
    fetch_10k,
    html_to_text,
    load_ticker_cik_map,
    parse_acceptance_datetime,
    pick_fy2025_10k,
)


NY = ZoneInfo("America/New_York")


def _make_recent(entries):
    """
    Build a filings.recent dict from a list of per-filing dicts, since
    the SEC payload uses parallel arrays rather than a list of objects.
    """
    keys = [
        "form",
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "primaryDocument",
    ]

    recent = {key: [] for key in keys}

    for entry in entries:
        for key in keys:
            recent[key].append(entry.get(key, ""))

    return {"filings": {"recent": recent}}


def test_load_ticker_cik_map_pads_to_10_digits():
    raw = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
    }

    mapping = load_ticker_cik_map(raw)

    assert mapping["AAPL"] == "0000320193"
    assert mapping["MSFT"] == "0000789019"


def test_load_ticker_cik_map_uppercases_ticker():
    raw = {"0": {"cik_str": 1, "ticker": "aapl", "title": "Apple"}}

    mapping = load_ticker_cik_map(raw)

    assert "AAPL" in mapping
    assert "aapl" not in mapping


def test_load_ticker_cik_map_skips_malformed_rows():
    raw = {
        "0": {"cik_str": 320193, "ticker": "AAPL"},
        "1": {"ticker": "NOCIK"},
        "2": {"cik_str": 12345},
    }

    mapping = load_ticker_cik_map(raw)

    assert mapping == {"AAPL": "0000320193"}


def test_parse_acceptance_datetime_treats_z_as_eastern():
    result = parse_acceptance_datetime("2025-10-31T18:01:50.000Z")

    assert result.hour == 18
    assert result.minute == 1
    assert result.tzinfo.key == "America/New_York"


def test_parse_acceptance_datetime_returns_tzaware():
    result = parse_acceptance_datetime("2026-02-15T09:00:00.000Z")

    assert result.tzinfo is not None
    assert result == datetime(2026, 2, 15, 9, 0, tzinfo=NY)


def test_parse_acceptance_datetime_without_fractional_seconds():
    result = parse_acceptance_datetime("2025-10-31T18:01:50Z")

    assert result.hour == 18
    assert result.tzinfo.key == "America/New_York"


def test_pick_fy2025_10k_calendar_year_filer():
    submissions = _make_recent([
        {
            "form": "10-K",
            "accessionNumber": "0000000001-26-000001",
            "filingDate": "2026-02-15",
            "reportDate": "2025-12-31",
            "acceptanceDateTime": "2026-02-15T08:00:00.000Z",
            "primaryDocument": "calendar-10k.htm",
        },
    ])

    selected = pick_fy2025_10k(submissions)

    assert selected["accession_number"] == "0000000001-26-000001"
    assert selected["filing_date"] == date(2026, 2, 15)
    assert selected["report_date"] == date(2025, 12, 31)


def test_pick_fy2025_10k_apple_fiscal_sep():
    submissions = _make_recent([
        {
            "form": "10-K",
            "accessionNumber": "0000320193-25-000123",
            "filingDate": "2025-10-31",
            "reportDate": "2025-09-27",
            "acceptanceDateTime": "2025-10-31T18:01:50.000Z",
            "primaryDocument": "aapl-20250927.htm",
        },
    ])

    selected = pick_fy2025_10k(submissions)

    assert selected["accession_number"] == "0000320193-25-000123"
    assert selected["report_date"] == date(2025, 9, 27)


def test_pick_fy2025_10k_excludes_amendments():
    submissions = _make_recent([
        {
            "form": "10-K/A",
            "accessionNumber": "0000000001-26-000002",
            "filingDate": "2026-03-01",
            "reportDate": "2025-12-31",
            "acceptanceDateTime": "2026-03-01T08:00:00.000Z",
            "primaryDocument": "amendment.htm",
        },
        {
            "form": "10-K",
            "accessionNumber": "0000000001-26-000001",
            "filingDate": "2026-02-15",
            "reportDate": "2025-12-31",
            "acceptanceDateTime": "2026-02-15T08:00:00.000Z",
            "primaryDocument": "original.htm",
        },
    ])

    selected = pick_fy2025_10k(submissions)

    assert selected["form"] == "10-K"
    assert selected["primary_document"] == "original.htm"


def test_pick_fy2025_10k_picks_latest_report_date():
    submissions = _make_recent([
        {
            "form": "10-K",
            "accessionNumber": "EARLIER",
            "filingDate": "2025-02-15",
            "reportDate": "2025-01-31",
            "acceptanceDateTime": "2025-02-15T08:00:00.000Z",
            "primaryDocument": "earlier.htm",
        },
        {
            "form": "10-K",
            "accessionNumber": "LATER",
            "filingDate": "2026-02-15",
            "reportDate": "2025-12-31",
            "acceptanceDateTime": "2026-02-15T08:00:00.000Z",
            "primaryDocument": "later.htm",
        },
    ])

    selected = pick_fy2025_10k(submissions)

    assert selected["accession_number"] == "LATER"


def test_pick_fy2025_10k_fiscal_january_filer():
    """
    NVDA / WMT / HD / CRM style: fiscal year ends late Jan or early
    Feb of 2025, 10-K filed in Feb-Mar 2025 - well before any
    mid-year filing window. The reportDate year is still 2025.
    """
    submissions = _make_recent([
        {
            "form": "10-K",
            "accessionNumber": "FY24",
            "filingDate": "2024-02-21",
            "reportDate": "2024-01-28",
            "acceptanceDateTime": "2024-02-21T17:00:00.000Z",
            "primaryDocument": "fy24.htm",
        },
        {
            "form": "10-K",
            "accessionNumber": "FY25",
            "filingDate": "2025-02-26",
            "reportDate": "2025-01-26",
            "acceptanceDateTime": "2025-02-26T17:30:00.000Z",
            "primaryDocument": "fy25.htm",
        },
    ])

    selected = pick_fy2025_10k(submissions)

    assert selected["accession_number"] == "FY25"
    assert selected["filing_date"] == date(2025, 2, 26)


def test_pick_fy2025_10k_skips_future_report_dates():
    submissions = _make_recent([
        {
            "form": "10-K",
            "accessionNumber": "FUTURE",
            "filingDate": "2027-02-15",
            "reportDate": "2026-12-31",
            "acceptanceDateTime": "2027-02-15T08:00:00.000Z",
            "primaryDocument": "future.htm",
        },
    ])

    with pytest.raises(LookupError):
        pick_fy2025_10k(submissions)


def test_pick_fy2025_10k_raises_when_missing():
    submissions = _make_recent([
        {
            "form": "10-Q",
            "accessionNumber": "0000000001-26-000005",
            "filingDate": "2026-04-01",
            "reportDate": "2026-03-31",
            "acceptanceDateTime": "2026-04-01T08:00:00.000Z",
            "primaryDocument": "10q.htm",
        },
    ])

    with pytest.raises(LookupError):
        pick_fy2025_10k(submissions)


def test_build_primary_doc_url_strips_dashes_from_accession():
    url = build_primary_doc_url(
        "0000320193", "0000320193-25-000123", "aapl-20250927.htm"
    )

    assert url == (
        "https://www.sec.gov/Archives/edgar/data/"
        "320193/000032019325000123/aapl-20250927.htm"
    )


def test_build_primary_doc_url_uses_integer_cik():
    url = build_primary_doc_url(
        "0000000001", "0000000001-26-000001", "doc.htm"
    )

    assert "/data/1/" in url


def test_html_to_text_removes_scripts_and_styles():
    html = (
        "<html><head><style>p {color: red;}</style></head>"
        "<body><script>alert('x')</script>"
        "<p>Visible content</p></body></html>"
    )

    text = html_to_text(html)

    assert "Visible content" in text
    assert "alert" not in text
    assert "color: red" not in text


def test_html_to_text_collapses_whitespace():
    html = "<p>Some    text    with\n\n\tlots   of   spaces.</p>"

    text = html_to_text(html)

    assert text == "Some text with lots of spaces."


def test_html_to_text_preserves_paragraph_breaks():
    html = "<p>Paragraph one.</p><p>Paragraph two.</p>"

    text = html_to_text(html)

    assert "Paragraph one." in text
    assert "Paragraph two." in text
    assert "\n\n" in text


def test_html_to_text_unwraps_inline_xbrl():
    html = (
        "<p>Revenue was "
        "<ix:nonFraction>123,456</ix:nonFraction>"
        " for the year.</p>"
    )

    text = html_to_text(html)

    assert "Revenue was 123,456 for the year." in text


class _FakeResponse:
    """
    Minimal stand-in for requests.Response covering the methods used.
    """

    def __init__(self, *, json_data=None, text=""):
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        return None


def _install_fake_get(monkeypatch, url_to_response, call_log=None):
    """
    Replace _sec_get with a lookup against a URL -> response map.
    """
    import edgar_retrieval

    def fake_get(url, user_agent):
        if call_log is not None:
            call_log.append(url)

        for matcher, response in url_to_response.items():
            if matcher in url:
                return response

        raise AssertionError(f"Unexpected SEC URL in test: {url}")

    monkeypatch.setattr(edgar_retrieval, "_sec_get", fake_get)


def _ticker_response():
    return _FakeResponse(
        json_data={
            "0": {
                "cik_str": 320193,
                "ticker": "AAPL",
                "title": "Apple Inc.",
            }
        }
    )


def _submissions_response():
    return _FakeResponse(
        json_data={
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "accessionNumber": ["0000320193-25-000123"],
                    "filingDate": ["2025-10-31"],
                    "reportDate": ["2025-09-27"],
                    "acceptanceDateTime": ["2025-10-31T18:01:50.000Z"],
                    "primaryDocument": ["aapl-20250927.htm"],
                }
            }
        }
    )


def _html_response():
    return _FakeResponse(
        text=(
            "<html><body><p>Item 1A. Risk Factors.</p>"
            "<p>The company faces competitive risks.</p></body></html>"
        )
    )


def test_fetch_10k_returns_full_shape(monkeypatch, tmp_path):
    _install_fake_get(monkeypatch, {
        "company_tickers.json": _ticker_response(),
        "submissions/CIK": _submissions_response(),
        "Archives/edgar/data": _html_response(),
    })

    result = fetch_10k("AAPL", user_agent="Test Agent test@example.com",
                       cache_dir=tmp_path)

    assert result["ticker"] == "AAPL"
    assert result["cik"] == "0000320193"
    assert result["accession_number"] == "0000320193-25-000123"
    assert result["form"] == "10-K"
    assert result["filing_date"] == date(2025, 10, 31)
    assert result["report_date"] == date(2025, 9, 27)
    assert result["primary_document"] == "aapl-20250927.htm"
    assert "320193" in result["primary_document_url"]
    assert "Risk Factors" in result["text"]
    assert Path(result["html_path"]).exists()
    assert Path(result["text_path"]).exists()


def test_fetch_10k_filing_timestamp_is_eastern_tzaware(monkeypatch, tmp_path):
    _install_fake_get(monkeypatch, {
        "company_tickers.json": _ticker_response(),
        "submissions/CIK": _submissions_response(),
        "Archives/edgar/data": _html_response(),
    })

    result = fetch_10k("AAPL", user_agent="Test", cache_dir=tmp_path)
    ts = result["filing_timestamp_et"]

    assert ts.tzinfo.key == "America/New_York"
    assert ts.hour == 18
    assert ts.minute == 1


def test_fetch_10k_uses_cache_on_second_call(monkeypatch, tmp_path):
    call_log = []

    _install_fake_get(monkeypatch, {
        "company_tickers.json": _ticker_response(),
        "submissions/CIK": _submissions_response(),
        "Archives/edgar/data": _html_response(),
    }, call_log=call_log)

    first = fetch_10k("AAPL", user_agent="Test", cache_dir=tmp_path)
    first_calls = len(call_log)

    second = fetch_10k("AAPL", user_agent="Test", cache_dir=tmp_path)
    second_calls = len(call_log) - first_calls

    assert second["accession_number"] == first["accession_number"]
    assert second["text"] == first["text"]
    assert second["filing_timestamp_et"] == first["filing_timestamp_et"]
    assert second_calls <= 2  # only the lookup endpoints, no HTML re-download


def test_fetch_10k_writes_meta_with_iso_strings(monkeypatch, tmp_path):
    _install_fake_get(monkeypatch, {
        "company_tickers.json": _ticker_response(),
        "submissions/CIK": _submissions_response(),
        "Archives/edgar/data": _html_response(),
    })

    fetch_10k("AAPL", user_agent="Test", cache_dir=tmp_path)

    meta_path = tmp_path / "AAPL" / "0000320193-25-000123.meta.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["ticker"] == "AAPL"
    assert meta["filing_date"] == "2025-10-31"
    assert meta["report_date"] == "2025-09-27"
    assert meta["filing_timestamp_et"] == "2025-10-31T18:01:50.000Z"


def test_fetch_10k_compatible_with_compute_t0(monkeypatch, tmp_path):
    """
    Cross-module smoke test: the filing_timestamp_et returned by
    fetch_10k must plug directly into t0_logic.compute_t0.
    """
    pytest.importorskip("pandas_market_calendars")

    _install_fake_get(monkeypatch, {
        "company_tickers.json": _ticker_response(),
        "submissions/CIK": _submissions_response(),
        "Archives/edgar/data": _html_response(),
    })

    result = fetch_10k("AAPL", user_agent="Test", cache_dir=tmp_path)

    t0_path = str(Path(__file__).parent.parent / "t0_logic")

    if t0_path not in sys.path:
        sys.path.insert(0, t0_path)

    from t0_logic import compute_t0

    t0 = compute_t0(result["filing_timestamp_et"])

    assert "t0_date" in t0
    assert "cutoff_timestamp_et" in t0
    assert "target_date" in t0
    assert t0["t0_date"] > result["filing_date"]


def test_dow30_constant_has_30_unique_tickers():
    assert len(DOW_30) == 30
    assert len(set(DOW_30)) == 30
    assert all(t.isupper() for t in DOW_30)
