import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

DEFAULT_SUBREDDITS = (
    "stocks",
    "investing",
    "wallstreetbets",
    "StockMarket",
)

# User-Agent sent on the no-auth JSON requests. Reddit throttles/blocks
# requests with a generic or empty UA, so this is used when the caller did
# not configure REDDIT_USER_AGENT.
DEFAULT_USER_AGENT = "finai-research/0.1 (academic; no-auth json)"

# Where the on-disk cache lives by default. Callers pass
# cache_dir=DEFAULT_POSTS_CACHE to turn caching on; tests pass tmp_path.
DEFAULT_POSTS_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "reddit"

# Submission fields captured at the network boundary. These are the names
# format_post reads, so a cached raw dict feeds straight back through it.
_RAW_FIELDS = (
    "id",
    "title",
    "selftext",
    "subreddit",
    "score",
    "num_comments",
    "url",
    "permalink",
    "created_utc",
)


def to_ny_time(ts: datetime) -> datetime:
    """
    Normalize timestamps to America/New_York.

    Assumes naive timestamps are already Eastern Time.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=NY_TZ)

    return ts.astimezone(NY_TZ)


def unix_to_ny_time(unix_timestamp) -> datetime:
    """
    Convert a Reddit (PRAW) created_utc UNIX timestamp to Eastern time.
    """
    return datetime.fromtimestamp(
        float(unix_timestamp),
        tz=UTC_TZ,
    ).astimezone(NY_TZ)


def _get(item, key):
    """
    Read a field from either a dict (test fixtures) or a PRAW submission
    object (live runs), returning None if absent.
    """
    if isinstance(item, dict):
        return item.get(key)

    return getattr(item, key, None)


def format_post(item):
    """
    Convert a raw Reddit submission into the project social-post format.

    Returns None if the item has no valid created_utc, since a post that
    cannot be timestamped cannot be checked against the cutoff and must
    not be allowed through.
    """
    try:
        created_utc = _get(item, "created_utc")

        if created_utc is None:
            return None

        published_at = unix_to_ny_time(created_utc)

    except (TypeError, ValueError, OSError):
        return None

    subreddit = _get(item, "subreddit")

    return {
        "id": _get(item, "id"),
        "title": _get(item, "title"),
        "body": _get(item, "selftext"),
        "subreddit": str(subreddit) if subreddit is not None else None,
        "score": _get(item, "score"),
        "num_comments": _get(item, "num_comments"),
        "url": _get(item, "url"),
        "permalink": _get(item, "permalink"),
        "published_at_et": published_at,
        "published_unix": created_utc,
    }


def filter_posts_before_cutoff(posts, cutoff_timestamp):
    """
    Keep only posts created at or before the T0 cutoff timestamp.

    This is the look-ahead leakage guard: Reddit search cannot constrain
    results to a precise instant, so every post is filtered here against
    the T0 close before the sentiment agent ever sees it.
    """
    cutoff_timestamp = to_ny_time(cutoff_timestamp)

    filtered = []

    for item in posts:
        formatted_item = format_post(item)

        if formatted_item is None:
            continue

        if formatted_item["published_at_et"] <= cutoff_timestamp:
            filtered.append(formatted_item)

    return filtered


def sort_posts_newest_first(posts):
    """
    Sort posts by publish time, newest first.
    """
    return sorted(
        posts,
        key=lambda item: item["published_at_et"],
        reverse=True,
    )


def fetch_reddit_posts(
    ticker,
    cutoff_timestamp,
    client_id=None,
    client_secret=None,
    user_agent=None,
    query=None,
    subreddits=DEFAULT_SUBREDDITS,
    limit=50,
    cache_dir=None,
    backend="auto",
):
    """
    Fetch Reddit posts about a company and filter them to prevent leakage.

    Inputs:
        ticker:          stock ticker, e.g. "AAPL".
        cutoff_timestamp: the T0 close timestamp from t0_logic.
        client_id / client_secret / user_agent:
                         Reddit API (PRAW) credentials. client_id /
                         client_secret are only needed for the "praw"
                         backend; user_agent is sent on the no-auth "json"
                         backend too (recommended, but defaulted if unset).
        query:           search query; defaults to the ticker symbol.
        subreddits:      finance subreddits to search.
        limit:           max submissions per subreddit.
        cache_dir:       optional directory for an on-disk request cache.
                         When set, a repeated search (same query +
                         subreddits + limit) is served from disk instead of
                         calling the Reddit API (which is rate limited).
                         Pass DEFAULT_POSTS_CACHE in production; None
                         disables caching (used by tests / offline paths).
        backend:         "praw"  - authenticated PRAW (needs client_id +
                                   client_secret);
                         "json"  - Reddit's public no-auth search.json
                                   endpoint (no credentials needed);
                         "auto"  - PRAW when both credentials are present,
                                   otherwise the no-auth JSON endpoint.

    Output:
        list of post dicts published at or before the cutoff, newest-first.
    """
    cutoff_timestamp = to_ny_time(cutoff_timestamp)
    search_query = query or ticker

    if backend == "auto":
        backend = "praw" if (client_id and client_secret) else "json"

    # The cache stores the RAW search results (plain dicts), which is what
    # costs API quota. The cutoff filter and sort are reapplied on every
    # read, so a different cutoff reuses the same cache and the leakage
    # guard is unchanged. Reddit search is not date-bounded; the request
    # signature is query + subreddits + limit. Mirrors the EDGAR / RAG /
    # FinnHub cache convention.
    raw_posts = None

    if cache_dir is not None:
        raw_posts = _read_posts_cache(
            cache_dir, ticker, search_query, subreddits, limit
        )

    # An empty cached result is treated as a miss. An empty Reddit search is
    # usually a transient block (e.g. HTTP 403 to the no-auth JSON endpoint
    # from a datacenter IP) rather than a genuine "no posts", so we neither
    # trust a cached empty list nor persist a freshly-fetched one. A later run
    # - with credentials, or from an unblocked IP - therefore refetches
    # instead of being stuck on an empty cache entry.
    if not raw_posts:
        if backend == "json":
            raw_posts = _reddit_search_json(
                search_query, subreddits, limit,
                user_agent or DEFAULT_USER_AGENT,
            )
        else:
            raw_posts = _reddit_search(
                search_query, subreddits, limit,
                client_id, client_secret, user_agent,
            )

        if cache_dir is not None and raw_posts:
            _write_posts_cache(
                cache_dir, ticker, search_query, subreddits, limit, raw_posts
            )

    # format_post reads dicts via _get, so the cached raw dicts feed back
    # through the same filter/sort path unchanged.
    filtered_posts = filter_posts_before_cutoff(raw_posts, cutoff_timestamp)

    return sort_posts_newest_first(filtered_posts)


def _reddit_search(query, subreddits, limit, client_id, client_secret, user_agent):
    """
    Single network boundary for Reddit. Tests monkeypatch this.

    Searches each subreddit and returns a list of JSON-serializable raw
    dicts (the fields format_post needs). Returning plain dicts rather than
    PRAW objects is what lets the result be cached to disk.
    """
    import praw

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    raw_posts = []

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)

        for submission in subreddit.search(query, sort="new", limit=limit):
            raw_posts.append({
                field: _raw_value(submission, field) for field in _RAW_FIELDS
            })

    return raw_posts


def _raw_value(submission, field):
    """
    Read one field off a submission for caching. `subreddit` is coerced to
    its name string (the PRAW Subreddit object is not serializable).
    """
    value = _get(submission, field)

    if field == "subreddit" and value is not None:
        return str(value)

    return value


def _http_get_json(url, user_agent):
    """
    Single HTTP boundary for the no-auth JSON backend. Tests monkeypatch
    this. Returns the parsed JSON body of a GET request.
    """
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _reddit_search_json(query, subreddits, limit, user_agent):
    """
    No-auth network boundary: Reddit's public search.json endpoint.

    Searches each subreddit via https://www.reddit.com/r/<sub>/search.json
    and maps the response to the same raw dict shape `_reddit_search`
    returns (the `_RAW_FIELDS` format_post reads), so the cache, cutoff
    filter, and sort downstream are identical regardless of backend.

    Requires no API credentials. Reddit rate-limits unauthenticated
    requests, so this is best paired with cache_dir. A subreddit that
    errors (rate limit, private, network) is skipped rather than aborting
    the whole run.
    """
    raw_posts = []

    for subreddit_name in subreddits:
        params = urllib.parse.urlencode({
            "q": query,
            "restrict_sr": "1",
            "sort": "new",
            "limit": limit,
        })
        url = f"https://www.reddit.com/r/{subreddit_name}/search.json?{params}"

        try:
            payload = _http_get_json(url, user_agent)
        except Exception:  # noqa: BLE001 - one bad subreddit must not abort
            continue

        children = (payload.get("data", {}) or {}).get("children", []) or []

        for child in children:
            data = child.get("data", {}) or {}
            raw_posts.append({
                field: data.get(field) for field in _RAW_FIELDS
            })

    return raw_posts


def _atomic_write(path, content):
    """
    Write content to path via a temporary file + rename, so partial writes
    from interrupted runs do not corrupt the cache.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    tmp_path.replace(path)


def _request_hash(query, subreddits, limit):
    """
    Short stable hash of the request signature, for the cache filename.
    """
    signature = f"{query}|{','.join(subreddits)}|{limit}"
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]


def _posts_cache_path(cache_dir, ticker, query, subreddits, limit):
    """
    Cache file for one search request:
    <cache_dir>/<TICKER>/posts_<hash>.json
    """
    digest = _request_hash(query, subreddits, limit)
    return Path(cache_dir) / ticker.upper() / f"posts_{digest}.json"


def _read_posts_cache(cache_dir, ticker, query, subreddits, limit):
    """
    Return the cached raw post dicts for this request, or None if absent.
    """
    path = _posts_cache_path(cache_dir, ticker, query, subreddits, limit)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("raw", [])


def _write_posts_cache(cache_dir, ticker, query, subreddits, limit, raw):
    """
    Atomically persist the raw search results for this request.
    """
    path = _posts_cache_path(cache_dir, ticker, query, subreddits, limit)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "ticker": ticker.upper(),
        "query": query,
        "subreddits": list(subreddits),
        "limit": limit,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw": raw,
    }

    _atomic_write(path, json.dumps(payload, indent=2))
