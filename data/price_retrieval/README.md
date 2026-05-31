# Price Data Retrieval (yfinance)

This module implements Task 3: stock price retrieval via yfinance.

It supplies the three price quantities the rest of the project needs:

    - baseline price        the close on T0
    - target price          the close on the 5th trading day after T0
    - pre-release trend      the closes for the trailing trading days up
                            to and including T0

The module takes:

    - ticker
    - T0 date            (from t0_logic.compute_t0)
    - target date        (from t0_logic.compute_t0)
    - optional trend_days (default 30)

and returns a single dict with the baseline price, target price, and
pre-release trend.

T0 and the target date both come from the Task 2 T0 computation module.
Keeping price lookup separate from the calendar/timing logic is a
deliberate split (see t0_logic): this module never decides *which* dates
matter, it only fetches the closes for the dates it is handed.

## No look-ahead leakage

The pre-release trend ends exactly at T0 and never includes a later
date. T0's close is both the baseline and the information cutoff, so a
trend ending at T0 carries no forward information. `get_price_trend`
filters to dates `<= t0_date` before selecting the trailing window.

The target price is fetched too, but it is the *answer key* used only by
the evaluation code (Tasks 8/9) - it is never part of what an agent
sees.

## Steps

1. Compute a download window

   `fetch_prices` spans from a calendar buffer before T0 (enough to
   cover `trend_days` trading days through weekends and holidays)
   through the target date.

2. Download daily closes (the single network boundary)

   `_download_close_series` calls `yfinance.download` with
   `auto_adjust=False`, so the returned close is the actual quoted close
   that evaluation compares against - not a split/dividend-adjusted
   series. It returns a plain `dict[date, float]`, which keeps all the
   selection logic pure and trivially testable.

3. Select the three quantities

   `get_close_on_date` looks up T0 and the target date (raising
   LookupError if the data genuinely does not cover a date), and
   `get_price_trend` returns the trailing window ending at T0.

The output of `fetch_prices` is a dict:

    {
        "ticker": str,
        "t0_date": date,
        "target_date": date,
        "baseline_price": float,
        "target_price": float,
        "pre_release_trend": [{"date": date, "close": float}, ...],
    }

## Usage

### 1. Install dependencies

This module is part of the unified uv project at the repo root:

    uv sync

### 2. Run the tests

    uv run pytest data/price_retrieval

All tests run offline: the network boundary `_download_close_series` is
monkeypatched with deterministic mock series, so no real yfinance call
is made.

### 3. Run against real data

    from datetime import date
    from price_retrieval import fetch_prices

    prices = fetch_prices("AAPL", date(2025, 11, 3), date(2025, 11, 10))
    print(prices["baseline_price"], prices["target_price"])

All cached Dow-30 filings at once (computes T0 from the EDGAR cache via
t0_logic, then fetches prices for each):

    uv run python data/price_retrieval/run_fetch_prices.py

The runner reads the EDGAR cache populated by
`data/EDGAR_retrieval/run_fetch.py`, computes T0 / target dates with
`t0_logic`, fetches prices for each ticker, and prints one line per
filing. Per-ticker failures do not stop the run.
