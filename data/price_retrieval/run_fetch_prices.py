"""
Fetch baseline / target prices and pre-release trend for every 10-K
already cached by the EDGAR module.

For each cached filing this reads the filing timestamp, computes T0 and
the target date with t0_logic, and fetches prices with yfinance.

Inputs : data/EDGAR_retrieval/cache/<TICKER>/<accession>.meta.json
Outputs: printed summary (one line per filing)

Per-ticker failures do not stop the run; a summary is printed at the end
and the script exits non-zero if anything failed.

Usage from the repo root:
    uv run python data/price_retrieval/run_fetch_prices.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "t0_logic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "EDGAR_retrieval"))

from price_retrieval import DEFAULT_PRICES_CACHE, fetch_prices  # noqa: E402
from t0_logic import compute_t0  # noqa: E402
from edgar_retrieval import DEFAULT_FILING_CACHE, parse_acceptance_datetime  # noqa: E402

EDGAR_CACHE = DEFAULT_FILING_CACHE


def _discover_filings():
    """
    Yield (ticker, filing_timestamp_et) for every cached EDGAR filing.
    """
    for ticker_dir in sorted(EDGAR_CACHE.iterdir()):
        if not ticker_dir.is_dir():
            continue

        for meta_path in sorted(ticker_dir.glob("*.meta.json")):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            yield (
                meta["ticker"],
                parse_acceptance_datetime(meta["filing_timestamp_et"]),
            )


def main():
    if not EDGAR_CACHE.exists():
        print(f"No EDGAR cache found at {EDGAR_CACHE}.")
        print("Run data/EDGAR_retrieval/run_fetch.py first.")
        sys.exit(1)

    filings = list(_discover_filings())

    if not filings:
        print(f"No cached filings under {EDGAR_CACHE}.")
        sys.exit(1)

    successes = []
    failures = []

    started = time.monotonic()

    for i, (ticker, filing_ts) in enumerate(filings, start=1):
        print(f"[{i:2d}/{len(filings)}] {ticker} ... ", end="", flush=True)

        try:
            dates = compute_t0(filing_ts)
            prices = fetch_prices(
                ticker,
                dates["t0_date"],
                dates["target_date"],
                cache_dir=DEFAULT_PRICES_CACHE,
            )

            print(
                f"T0={dates['t0_date']} ${prices['baseline_price']:.2f}  "
                f"target={dates['target_date']} ${prices['target_price']:.2f}"
            )

            successes.append((ticker, prices))

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
