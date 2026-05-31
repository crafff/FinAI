# EDGAR 10-K Retrieval

This module implements Task 1: SEC EDGAR 10-K retrieval. It is the
prerequisite data step for the whole project.

The module takes:

    - ticker
    - SEC User-Agent string (name and email)
    - optional cache directory
    - optional fiscal year (default 2025)

and returns the company’s FY2025 10-K together with its filing
timestamp, so downstream modules can compute T₀ and feed the text into
the RAG pipeline.

The filing timestamp returned by this module is meant to flow directly
into the Task 2 T₀ computation module. In the project protocol, T₀ and
the information cutoff are derived from that timestamp; an incorrect
filing time would propagate as look-ahead leakage everywhere else.

The module performs six main steps:

1. Look up CIK

   The SEC company_tickers.json file is downloaded and parsed into a
   ticker → 10-digit zero-padded CIK mapping.

2. Read submissions metadata

   The data.sec.gov submissions endpoint is called with the padded
   CIK. Its payload contains parallel arrays of recent filings with
   form type, filing date, report date, acceptance datetime, and
   primary document.

3. Select the FY2025 10-K

   From filings.recent the module keeps entries where:

       - form is exactly "10-K"     (10-K/A amendments are excluded)
       - reportDate.year equals the fiscal year

   Among candidates the one with the latest reportDate is chosen.
   reportDate alone identifies the fiscal year for every convention:

       - fiscal-January filers (NVDA, WMT, HD, CRM):
             reportDate ~2025-01-XX, filed Feb-Mar 2025
       - fiscal-June filers (MSFT):
             reportDate 2025-06-30, filed Jul-Aug 2025
       - fiscal-September filers (AAPL):
             reportDate 2025-09-27, filed Oct-Nov 2025
       - calendar-year filers (most others):
             reportDate 2025-12-31, filed Jan-Mar 2026

   A filingDate window would wrongly exclude the fiscal-January
   cohort, so the module does not use one.

4. Parse the filing timestamp

   SEC serializes acceptanceDateTime as "2025-10-31T18:01:50.000Z",
   but the value is Eastern Time despite the trailing Z. The module
   strips the Z, parses naive, and attaches America/New_York. The
   resulting datetime plugs into t0_logic.compute_t0 directly.

5. Download and clean the primary document

   The 10-K primary HTML document is downloaded from Archives. The
   HTML is also converted to plain text: script and style blocks are
   removed, inline-XBRL tags are unwrapped, and whitespace is collapsed
   while paragraph breaks are preserved as blank lines.

6. Cache to disk

   When a cache directory is provided, each filing is written under
   <cache_dir>/<TICKER>/ as three files:

       - <accession>.html        the original primary document
       - <accession>.txt         the cleaned plain text
       - <accession>.meta.json   the metadata (dates as ISO strings)

   Writes go through a .tmp file then rename so an interrupted run
   cannot leave a partial cache entry. On the next call for the same
   ticker, no HTTP request to Archives is made.

The output of fetch_10k is a dict containing:

    - ticker
    - cik                    (10-digit zero-padded string)
    - accession_number
    - form
    - filing_date            (date object)
    - filing_timestamp_et    (NY-aware datetime, feeds compute_t0)
    - report_date            (date object, fiscal year end)
    - primary_document
    - primary_document_url
    - html_path
    - text_path
    - text                   (cleaned plain text of the 10-K)

A fetch_dow30_10ks convenience wrapper iterates over the DOW_30
constant and returns one result per ticker.

The SEC requires a descriptive User-Agent and limits requests to about
ten per second. The module enforces a small minimum interval between
requests and passes the caller-supplied User-Agent on every call.

The module is designed to be testable without network access. Unit
tests build mock SEC payloads, monkeypatch the single HTTP boundary
function, and verify ticker mapping, FY2025 selection, timestamp
parsing, URL construction, HTML cleaning, caching, and end-to-end
integration with compute_t0.

## Usage

### 1. Install dependencies

    cd data/EDGAR_retrieval
    pip install -r requirements.txt

requirements.txt pulls in requests, beautifulsoup4, lxml, and pytest.

### 2. Run the tests

    pytest -v

All 24 tests run offline. They build mock SEC payloads in memory and
monkeypatch the single HTTP boundary, so no network and no real
User-Agent are needed. The cross-module test against compute_t0 is
skipped automatically if pandas_market_calendars is not installed.

### 3. Run against real SEC data

Single ticker, from a Python REPL or script:

    from pathlib import Path
    import sys
    sys.path.insert(0, "../t0_logic")

    from edgar_retrieval import fetch_10k
    from t0_logic import compute_t0

    result = fetch_10k(
        "AAPL",
        user_agent="FinAI Research ruitaozhou2002@gmail.com",
        cache_dir=Path("./cache"),
    )

    print(result["filing_date"], result["filing_timestamp_et"])
    print("text length:", len(result["text"]))
    print("T0:", compute_t0(result["filing_timestamp_et"]))

Expected for AAPL FY2025: filing_date 2025-10-31, after-hours filing
timestamp, T₀ rolls to the next trading day, text length on the order
of a few hundred kilobytes.

All 30 Dow constituents at once:

    python run_fetch.py

The runner iterates over the DOW_30 constant, calls fetch_10k for
each ticker, writes html / text / meta json to ./cache/<TICKER>/, and
prints one progress line per filing. Per-ticker failures do not stop
the run; a summary prints at the end and the script exits non-zero if
anything failed. Already-cached filings return without HTTP traffic,
so reruns are fast.

The User-Agent is read from the `SEC_USER_AGENT` variable in your `.env`
(see `config/README.md`). SEC requires a descriptive identifier (name +
contact email); set it there before running.
