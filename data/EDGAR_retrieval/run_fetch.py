"""
Fetch FY2025 10-Ks for the 30 Dow Jones constituents to local cache.

Outputs go to ./cache/<TICKER>/ as three files per filing:
    <accession>.html, <accession>.txt, <accession>.meta.json

Per-ticker failures do not stop the run; a summary is printed at the
end and the script exits non-zero if anything failed.

The SEC User-Agent is read from the SEC_USER_AGENT variable in your .env
(see config/README.md). SEC requires a descriptive identifier (name +
contact email).

Usage from the repo root:
    uv run python data/EDGAR_retrieval/run_fetch.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "config"))

from edgar_retrieval import DOW_30, fetch_10k
from settings import MissingConfigError, load_settings  # noqa: E402


CACHE_DIR = Path(__file__).parent / "cache"


def main():
    try:
        user_agent = load_settings().require_sec_user_agent()
    except MissingConfigError as exc:
        print(exc)
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    successes = []
    failures = []

    started = time.monotonic()

    for i, ticker in enumerate(DOW_30, start=1):
        print(f"[{i:2d}/30] {ticker} ... ", end="", flush=True)

        try:
            result = fetch_10k(
                ticker,
                user_agent,
                cache_dir=CACHE_DIR,
            )

            print(
                f"{result['filing_date']}  "
                f"acc={result['accession_number']}  "
                f"text={len(result['text']):>9,} chars"
            )

            successes.append((ticker, result))

        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            failures.append((ticker, exc))

    elapsed = time.monotonic() - started

    print()
    print(
        f"Done in {elapsed:.1f}s: "
        f"{len(successes)} succeeded, {len(failures)} failed"
    )

    if failures:
        print()
        print("Failures:")

        for ticker, exc in failures:
            print(f"  {ticker}: {type(exc).__name__}: {exc}")

        sys.exit(1)


if __name__ == "__main__":
    main()
