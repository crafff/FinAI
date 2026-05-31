# Social Media Retrieval (Reddit / PRAW)

This module implements Task 6: company-related Reddit posts via PRAW,
with the same no-look-ahead cutoff logic as the FinnHub news module
(Task 5).

The module takes:

    - ticker
    - T0 cutoff timestamp     (from t0_logic.compute_t0)
    - Reddit API credentials  (client_id, client_secret, user_agent)
    - optional query (defaults to the ticker), subreddits, and limit

and returns Reddit posts created at or before the T0 cutoff.

The cutoff timestamp is the market close on T0. Posts created after that
cutoff must not be visible to the sentiment agent, because they would
contain information from the prediction window.

## Steps

1. Normalize timestamps

   All cutoff timestamps are converted to America/New_York. Naive
   timestamps are treated as already being Eastern Time. PRAW's
   `created_utc` is a UNIX timestamp, converted to Eastern via
   `unix_to_ny_time`.

2. Fetch raw posts

   `fetch_reddit_posts` searches each finance subreddit
   (default: stocks, investing, wallstreetbets, StockMarket) for the
   ticker, sorted newest-first, up to `limit` per subreddit.

3. Filter posts by cutoff

   Reddit search cannot constrain results to a precise instant, so every
   post is filtered against the cutoff:

       published_at_et <= cutoff_timestamp_et

   This is the most important step for preventing look-ahead leakage.
   Posts with missing or invalid `created_utc` are skipped because they
   cannot be safely compared against the cutoff.

4. Sort posts

   The filtered posts are sorted newest-first so the sentiment agent
   sees the most recent valid information first.

The output for each post includes:

    id, title, body, subreddit, score, num_comments, url, permalink,
    published_at_et, published_unix

`format_post` reads fields from either a live PRAW submission object
(attribute access) or a dict, so the parsing/filtering/sorting logic is
fully testable without the PRAW dependency or live credentials.

## Caching

The Reddit API is rate limited, so search results are cached to disk when a
`cache_dir` is given (mirrors the EDGAR / RAG / FinnHub convention). The
single network boundary `_reddit_search` returns JSON-serializable raw post
dicts; these are cached keyed by the search request (query + subreddits +
limit) at `<cache_dir>/<TICKER>/posts_<hash>.json`. The cutoff filter and
sort are reapplied on every read — Reddit search is not date-bounded, so a
different cutoff reuses the same cache and leakage protection is unchanged.
Pass `cache_dir=DEFAULT_POSTS_CACHE` (`data/reddit_retrieval/cache/`,
gitignored) in production; `None` disables caching (used by tests).

## Usage

### 1. Install dependencies

This module is part of the unified uv project at the repo root:

    uv sync

Reddit API access requires a registered app
(https://www.reddit.com/prefs/apps). The runner reads credentials from
the `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT`
environment variables.

### 2. Run the tests

    uv run pytest data/reddit_retrieval

All tests run offline using mock submissions (both dicts and a fake PRAW
object), so no credentials and no network are needed. They cover
timezone conversion, attribute/dict parsing, cutoff filtering, malformed
posts, and sorting.

### 3. Run against real data

    import os
    from datetime import datetime
    from reddit_retrieval import fetch_reddit_posts, DEFAULT_POSTS_CACHE

    posts = fetch_reddit_posts(
        "AAPL",
        cutoff_timestamp=datetime(2025, 11, 3, 16, 0),
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
        cache_dir=DEFAULT_POSTS_CACHE,  # repeated searches served from disk
    )
    print(len(posts), "posts before cutoff")
