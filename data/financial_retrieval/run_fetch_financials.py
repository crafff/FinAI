"""
Fetch structured fundamentals from FMP for the 30 Dow Jones
constituents.

Requires an FMP API key in the FMP_API_KEY environment variable.

Per-ticker failures do not stop the run; a summary is printed at the end
and the script exits non-zero if anything failed.

Usage from the repo root:
    FMP_API_KEY=... uv run python data/financial_retrieval/run_fetch_financials.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "EDGAR_retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "config"))


from financial_retrieval import (  # noqa: E402
    DEFAULT_FINANCIALS_CACHE,
    fetch_financials,
)
from edgar_retrieval import DOW_30  # noqa: E402
from settings import MissingConfigError, load_settings  # noqa: E402


def main():
    try:
        api_key = load_settings().require_fmp_api_key()
    except MissingConfigError as exc:
        print(exc)
        sys.exit(1)

    successes = []
    failures = []

    started = time.monotonic()

    for i, ticker in enumerate(DOW_30, start=1):
        print(f"[{i:2d}/30] {ticker} ... ", end="", flush=True)

        try:
            data = fetch_financials(
                ticker,
                api_key,
                cache_dir=DEFAULT_FINANCIALS_CACHE,
            )

            profitability = data["profitability"]
            revenue = profitability["revenue"]
            net_margin = profitability["net_margin"]

            revenue_str = f"${revenue:,.0f}" if revenue is not None else "n/a"
            margin_str = (
                f"{net_margin:.1%}" if net_margin is not None else "n/a"
            )

            print(
                f"FY{data['fiscal_year']}  "
                f"revenue={revenue_str}  net_margin={margin_str}"
            )

            successes.append((ticker, data))

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
