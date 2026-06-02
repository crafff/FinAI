import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data/"
    "{cik_int}/{acc_no_dashes}/{primary_doc}"
)

SEC_MIN_REQUEST_INTERVAL_SECONDS = 0.11

# Where the on-disk 10-K cache lives by default. Callers pass
# cache_dir=DEFAULT_FILING_CACHE to turn caching on; tests pass tmp_path.
DEFAULT_FILING_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "edgar"

DOW_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA",
    "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW",
    "TRV", "UNH", "V", "VZ", "WMT",
]


_last_sec_request_time = [0.0]


def load_ticker_cik_map(raw_json):
    """
    Convert the SEC company_tickers.json payload into a mapping
    of uppercased ticker -> zero-padded 10-digit CIK string.

    The SEC payload is a dict whose values look like:
        {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    """
    mapping = {}

    for entry in raw_json.values():
        ticker = entry.get("ticker")
        cik = entry.get("cik_str")

        if ticker is None or cik is None:
            continue

        mapping[ticker.upper()] = str(int(cik)).zfill(10)

    return mapping


def parse_acceptance_datetime(value):
    """
    Parse SEC submissions.json acceptanceDateTime into a NY-aware datetime.

    SEC serializes this field like "2025-10-31T18:01:50.000Z", but the
    value is actually Eastern Time despite the trailing Z. We strip the
    Z, parse as naive, and attach NY_TZ.
    """
    if value is None:
        raise ValueError("acceptanceDateTime is missing.")

    cleaned = value.strip()

    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]

    if "." in cleaned:
        cleaned = cleaned.split(".", 1)[0]

    naive = datetime.fromisoformat(cleaned)

    return naive.replace(tzinfo=NY_TZ)


def _parse_date(value):
    """
    Parse a YYYY-MM-DD string into a date object.
    """
    return datetime.strptime(value, "%Y-%m-%d").date()


def pick_fy2025_10k(submissions_json, fiscal_year=2025):
    """
    Select the FY<fiscal_year> 10-K entry from a submissions.json payload.

    Walk filings.recent (parallel arrays). Keep entries where:
        - form == "10-K"               (excludes 10-K/A amendments)
        - reportDate.year == fiscal_year

    Among candidates, pick the one with the latest reportDate. The
    period-end year alone identifies the fiscal year for every
    convention: fiscal-January filers (NVDA, WMT, HD, CRM) have
    reportDate ~2025-01-XX and file in Feb-Mar 2025; fiscal-June (MSFT)
    files in July-Aug 2025 with reportDate 2025-06-30; fiscal-September
    (AAPL) files in Oct-Nov 2025 with reportDate 2025-09-27;
    calendar-year filers file in Jan-Mar 2026 with reportDate
    2025-12-31. A filingDate window would wrongly exclude the
    fiscal-January cohort.

    Returns a dict with raw SEC fields. Raises LookupError if none found.
    """
    recent = submissions_json.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    acceptance_datetimes = recent.get("acceptanceDateTime", [])
    primary_documents = recent.get("primaryDocument", [])

    candidates = []

    for i, form in enumerate(forms):
        if form != "10-K":
            continue

        report_date_str = report_dates[i] if i < len(report_dates) else ""
        filing_date_str = filing_dates[i] if i < len(filing_dates) else ""

        if not report_date_str or not filing_date_str:
            continue

        try:
            report_date = _parse_date(report_date_str)
            filing_date = _parse_date(filing_date_str)
        except ValueError:
            continue

        if report_date.year != fiscal_year:
            continue

        candidates.append({
            "form": form,
            "accession_number": accession_numbers[i],
            "filing_date": filing_date,
            "report_date": report_date,
            "acceptance_datetime": acceptance_datetimes[i],
            "primary_document": primary_documents[i],
        })

    if not candidates:
        raise LookupError(
            f"No FY{fiscal_year} 10-K found in submissions.recent."
        )

    candidates.sort(key=lambda c: c["report_date"], reverse=True)

    return candidates[0]


def build_primary_doc_url(cik, accession_number, primary_document):
    """
    Build the URL of the 10-K primary document on EDGAR Archives.

    EDGAR Archives uses the integer form of CIK and the accession
    number with dashes removed.
    """
    cik_int = int(cik)
    acc_no_dashes = accession_number.replace("-", "")

    return SEC_ARCHIVES_URL.format(
        cik_int=cik_int,
        acc_no_dashes=acc_no_dashes,
        primary_doc=primary_document,
    )


_BLOCK_TAGS = (
    "p", "div", "li", "tr", "br", "table",
    "h1", "h2", "h3", "h4", "h5", "h6",
)

_PARAGRAPH_SENTINEL = "\x00PARA\x00"


def html_to_text(html):
    """
    Convert SEC 10-K HTML into clean plain text.

    Strips <script>, <style>, and inline-XBRL <ix:*> tags. Collapses
    whitespace within each text node so embedded newlines inside a
    paragraph become a single space; preserves structural paragraph
    breaks between block-level elements as double newlines.
    """
    import warnings

    from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        tag.unwrap()

    for node in list(soup.find_all(string=True)):
        node.replace_with(re.sub(r"\s+", " ", str(node)))

    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_after(NavigableString(_PARAGRAPH_SENTINEL))

    text = soup.get_text()

    paragraphs = []

    for chunk in text.split(_PARAGRAPH_SENTINEL):
        chunk = chunk.strip()

        if chunk:
            paragraphs.append(chunk)

    return "\n\n".join(paragraphs)


def _sec_get(url, user_agent):
    """
    Single HTTP boundary for SEC requests. Tests monkeypatch this.

    Enforces the SEC rate limit (~10 req/s) with a module-level
    last-call timestamp and passes the required User-Agent header.
    """
    import requests

    elapsed = time.monotonic() - _last_sec_request_time[0]

    if elapsed < SEC_MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(SEC_MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": _host_from_url(url),
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    _last_sec_request_time[0] = time.monotonic()

    return response


def _host_from_url(url):
    """
    Extract host for the SEC Host header.
    """
    match = re.match(r"https?://([^/]+)/", url)

    if match is None:
        return ""

    return match.group(1)


def get_ticker_cik(ticker, user_agent):
    """
    Look up the zero-padded 10-digit CIK for a ticker via the SEC
    company_tickers.json endpoint.
    """
    response = _sec_get(SEC_TICKERS_URL, user_agent)
    mapping = load_ticker_cik_map(response.json())

    ticker_upper = ticker.upper()

    if ticker_upper not in mapping:
        raise LookupError(f"Ticker {ticker_upper} not found in SEC ticker map.")

    return mapping[ticker_upper]


def _read_cache(cache_dir, ticker, accession_number):
    """
    Return cached result dict if all three artifacts are present,
    otherwise None.
    """
    ticker_dir = cache_dir / ticker.upper()

    meta_path = ticker_dir / f"{accession_number}.meta.json"
    html_path = ticker_dir / f"{accession_number}.html"
    text_path = ticker_dir / f"{accession_number}.txt"

    if not (meta_path.exists() and html_path.exists() and text_path.exists()):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    return _hydrate_result(meta, text, html_path, text_path)


def _hydrate_result(meta, text, html_path, text_path):
    """
    Rehydrate stored metadata into the public return shape, restoring
    date / datetime types from their ISO string forms.
    """
    return {
        "ticker": meta["ticker"],
        "cik": meta["cik"],
        "accession_number": meta["accession_number"],
        "form": meta["form"],
        "filing_date": _parse_date(meta["filing_date"]),
        "filing_timestamp_et": parse_acceptance_datetime(
            meta["filing_timestamp_et"]
        ),
        "report_date": _parse_date(meta["report_date"]),
        "primary_document": meta["primary_document"],
        "primary_document_url": meta["primary_document_url"],
        "html_path": str(html_path),
        "text_path": str(text_path),
        "text": text,
    }


def _write_cache(
    cache_dir, ticker, accession_number, html, text, meta
):
    """
    Atomically write the HTML, text, and metadata for one filing.
    """
    ticker_dir = cache_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    html_path = ticker_dir / f"{accession_number}.html"
    text_path = ticker_dir / f"{accession_number}.txt"
    meta_path = ticker_dir / f"{accession_number}.meta.json"

    _atomic_write(html_path, html)
    _atomic_write(text_path, text)
    _atomic_write(meta_path, json.dumps(meta, indent=2))

    return html_path, text_path


def _atomic_write(path, content):
    """
    Write content to path via a temporary file + rename, so partial
    writes from interrupted runs do not corrupt the cache.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    tmp_path.replace(path)


def fetch_10k(ticker, user_agent, cache_dir=None, fiscal_year=2025):
    """
    Retrieve a company's FY<fiscal_year> 10-K from SEC EDGAR.

    Steps:
        1. Look up CIK via company_tickers.json.
        2. Read CIK<cik>.json from the submissions API.
        3. Select the FY<fiscal_year> 10-K (excludes amendments).
        4. Download the primary HTML document.
        5. Extract plain text.
        6. Optionally cache html, text, and metadata to disk.

    Returns a dict whose filing_timestamp_et field feeds directly into
    t0_logic.compute_t0 (see data/t0_logic/t0_logic.py).
    """
    ticker_upper = ticker.upper()

    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    cik = get_ticker_cik(ticker_upper, user_agent)

    submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
    submissions_response = _sec_get(submissions_url, user_agent)
    submissions_json = submissions_response.json()

    selected = pick_fy2025_10k(submissions_json, fiscal_year=fiscal_year)

    accession_number = selected["accession_number"]

    if cache_dir is not None:
        cached = _read_cache(cache_dir, ticker_upper, accession_number)

        if cached is not None:
            return cached

    primary_document_url = build_primary_doc_url(
        cik, accession_number, selected["primary_document"]
    )

    html_response = _sec_get(primary_document_url, user_agent)
    html = html_response.text
    text = html_to_text(html)

    filing_timestamp_et = parse_acceptance_datetime(
        selected["acceptance_datetime"]
    )

    meta = {
        "ticker": ticker_upper,
        "cik": cik,
        "accession_number": accession_number,
        "form": selected["form"],
        "filing_date": selected["filing_date"].isoformat(),
        "filing_timestamp_et": selected["acceptance_datetime"],
        "report_date": selected["report_date"].isoformat(),
        "primary_document": selected["primary_document"],
        "primary_document_url": primary_document_url,
    }

    html_path = None
    text_path = None

    if cache_dir is not None:
        html_path, text_path = _write_cache(
            cache_dir, ticker_upper, accession_number, html, text, meta
        )

    return {
        "ticker": ticker_upper,
        "cik": cik,
        "accession_number": accession_number,
        "form": selected["form"],
        "filing_date": selected["filing_date"],
        "filing_timestamp_et": filing_timestamp_et,
        "report_date": selected["report_date"],
        "primary_document": selected["primary_document"],
        "primary_document_url": primary_document_url,
        "html_path": str(html_path) if html_path else None,
        "text_path": str(text_path) if text_path else None,
        "text": text,
    }


def fetch_dow30_10ks(user_agent, cache_dir=None, fiscal_year=2025):
    """
    Fetch FY<fiscal_year> 10-Ks for all 30 Dow Jones constituents.

    Returns a list of result dicts in the same order as DOW_30.
    Individual lookup failures are not swallowed; the caller decides
    how to handle them by wrapping per-ticker if needed.
    """
    results = []

    for ticker in DOW_30:
        result = fetch_10k(
            ticker,
            user_agent,
            cache_dir=cache_dir,
            fiscal_year=fiscal_year,
        )
        results.append(result)

    return results

