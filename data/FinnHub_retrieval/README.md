# FinnHub News Retrieval

This module implements Task 5: company news retrieval using FinnHub.

The purpose of this module is to retrieve company-related news while
strictly enforcing the project’s no-look-ahead leakage rule.

The module takes:

    - ticker
    - T₀ cutoff timestamp
    - FinnHub API key
    - optional lookback window

and returns company news articles published at or before the T₀ cutoff.

The cutoff timestamp should come from the Task 2 T₀ computation module.
In the project protocol, the cutoff is the market close on T₀. News
published after that cutoff must not be visible to the sentiment agent,
because it would contain information from the prediction window.

The module performs four main steps:

1. Normalize timestamps

   All cutoff timestamps are converted to America/New_York time so that
   news timestamps and market cutoff logic are compared consistently.

   Naive timestamps are treated as already being in Eastern Time.

2. Fetch raw company news

   The module calls FinnHub’s company_news endpoint using a date range
   ending on the T₀ cutoff date.

3. Filter articles by cutoff

   Even though FinnHub requests use date ranges, the endpoint may return
   articles from the cutoff date after market close. Therefore, the code
   applies a second timestamp-level filter:

       published_at_et <= cutoff_timestamp_et

   This is the most important step for preventing look-ahead leakage.

   Articles with missing or invalid timestamps are skipped because they
   cannot be safely compared against the cutoff.

4. Sort articles

   The filtered articles are sorted newest-first so the sentiment agent
   sees the most recent valid information first.

The output for each article includes:

    - headline
    - summary
    - source
    - url
    - published_at_et
    - published_unix

This module is designed to be testable without a FinnHub API key. The
unit tests use mock news articles with artificial UNIX timestamps to
verify that cutoff filtering, timezone conversion, malformed article
handling, and sorting work correctly.

This makes Task 5 independent of the rest of the agent pipeline while
still aligning with the overall project requirement that all
agent-visible information must be cut off at the T₀ close.
