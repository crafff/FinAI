# Financial Data Retrieval (FMP)

This module implements Task 4: structured company fundamentals from the
Financial Modeling Prep (FMP) API.

The 10-K contains the same figures, but buried in long text and tables
where direct LLM extraction is error-prone. This module supplies clean,
typed numbers so the fundamental agent (Task 10) can reason over
reliable metrics, while the 10-K (via RAG) supplies the narrative and
context behind them.

The module takes:

    - ticker
    - FMP API key
    - optional as_of_fiscal_year (default 2025)
    - optional period ("annual") and limit

and returns one merged dict of profitability, cash-flow, debt, and
valuation metrics for the selected fiscal period.

## No look-ahead leakage

Two leakage concerns are handled explicitly:

1. Period selection. A run executed in 2026 may see a FY2026 statement
   that postdates the FY2025 10-K and its T0 cutoff.
   `select_latest_period` drops every period whose fiscal year is
   greater than `as_of_fiscal_year` before choosing the latest
   remaining one. So with the default `as_of_fiscal_year=2025`, only
   FY2025-and-earlier fundamentals are ever returned.

2. Valuation ratios. The ratios returned here are computed by FMP at the
   fiscal period end, so they reflect the *period-end* price, not a live
   price. They are safe as point-in-time context. Any valuation ratio
   that must reflect the prediction-moment price should instead be
   recomputed downstream from the T0 baseline close (see
   price_retrieval) - never pulled live, which would leak post-cutoff
   information.

## Steps

1. Fetch four statements (the single network boundary)

   `_fmp_get` calls the FMP `/stable` endpoints `income-statement`,
   `balance-sheet-statement`, `cash-flow-statement`, and `ratios`,
   returning each as a list of period dicts (newest first). It raises on
   an FMP error payload (bad symbol / quota).

2. Select the period

   `select_latest_period` applies the leakage guard and picks the latest
   eligible fiscal year, using `calendarYear` and falling back to the
   `date` field.

3. Merge into the project schema

   `merge_financials` maps the raw FMP fields into typed groups via
   `_to_float`, which coerces blanks/missing to None rather than raising.

The output of `fetch_financials` is a dict:

    {
        "ticker": str,
        "fiscal_year": int | None,
        "report_date": str | None,
        "profitability": {revenue, net_income, gross_margin,
                          operating_margin, net_margin,
                          return_on_equity, return_on_assets},
        "cash_flow":     {operating_cash_flow, capital_expenditure,
                          free_cash_flow},
        "debt":          {total_debt, total_equity, debt_to_equity,
                          current_ratio, interest_coverage},
        "valuation":     {pe_ratio, pb_ratio, price_to_sales,
                          ev_to_ebitda},
    }

## Usage

### 1. Install dependencies

This module is part of the unified uv project at the repo root:

    uv sync

FMP requires a free API key (https://financialmodelingprep.com). The
runner reads it from the `FMP_API_KEY` environment variable.

### 2. Run the tests

    uv run pytest data/financial_retrieval

All tests run offline: the network boundary `_fmp_get` is monkeypatched
with mock FMP payloads, so no API key and no network are needed. The
tests cover field mapping, the fiscal-year leakage guard, and end-to-end
merging.

### 3. Run against real data

    import os
    from financial_retrieval import fetch_financials, DEFAULT_FINANCIALS_CACHE

    data = fetch_financials(
        "AAPL",
        api_key=os.environ["FMP_API_KEY"],
        cache_dir=DEFAULT_FINANCIALS_CACHE,  # repeated requests served from disk
    )
    print(data["fiscal_year"], data["profitability"])

## Caching

Each `fetch_financials` call spends four FMP requests (income, balance,
cash-flow, ratios), and FMP's free tier is quota limited. When a `cache_dir`
is given, the four **raw** endpoint responses are cached to disk keyed by
`(ticker, period, limit)` at
`<cache_dir>/<TICKER>/financials_<period>_<limit>.json`; a repeated request is
served from disk and spends no quota. The leakage guard
(`select_latest_period` with `as_of_fiscal_year`) and the merge run on every
read, so a different `as_of_fiscal_year` reuses the same cache. Pass
`cache_dir=DEFAULT_FINANCIALS_CACHE` (`data/financial_retrieval/cache/`,
gitignored) in production; `None` disables caching (used by tests).

All Dow-30 tickers at once:

    FMP_API_KEY=... uv run python data/financial_retrieval/run_fetch_financials.py

The runner iterates the Dow-30 ticker list, calls `fetch_financials` for
each, and prints one summary line per company. Per-ticker failures do
not stop the run.
