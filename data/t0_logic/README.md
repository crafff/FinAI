This implementation follows the project specification’s
unified T₀ rule:

• If the filing occurs before market open
  (9:30 AM ET) on a valid NYSE trading day:
      T₀ = same trading day

• Otherwise (intraday, after-hours, weekend,
  or holiday filing):
      T₀ = next trading day

The implementation is responsible only for the
calendar and timing logic required for Task 2.
It does not retrieve stock prices directly.

The system performs four main steps:

1. Normalize timestamps
   All filing timestamps are converted to the
   America/New_York timezone so comparisons
   against NYSE market hours are consistent.

2. Determine valid trading days
   The implementation uses the NYSE trading
   calendar provided by pandas_market_calendars
   to correctly handle:
      - weekends
      - market holidays
      - trading-day boundaries
      - early-close sessions

3. Compute evaluation dates
   The module computes:
      - T₀ date
      - cutoff timestamp (market close on T₀)
      - target date (5th trading day after T₀)

4. Separate timing logic from price retrieval
   A separate module should later retrieve:
      - baseline price = close on T₀
      - target price = close on target date
   using yfinance or another market-data API.

The implementation correctly handles all major
edge cases required by the specification:
      - pre-market filings
      - intraday filings
      - after-hours filings
      - weekend filings
      - holiday filings

One subtle design decision involves filings
submitted exactly at 9:30 AM ET.

Because the specification states:
      “before market open”
rather than:
      “before or at market open”

a filing exactly at 9:30 AM ET is treated as
occurring after market open and therefore rolls
forward to the next trading day.

Additionally, the cutoff timestamp is derived
directly from the NYSE exchange schedule rather
than assuming a fixed 4:00 PM close. This keeps
the implementation correct on early-close trading
sessions such as Black Friday.
