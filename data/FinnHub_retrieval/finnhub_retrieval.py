import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Where the on-disk cache lives by default. Callers (pipeline / runners)
# pass cache_dir=DEFAULT_NEWS_CACHE to turn caching on; tests pass tmp_path.
DEFAULT_NEWS_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "finnhub"


def to_ny_time(ts: datetime) -> datetime:
    """
    Normalize timestamps to America/New_York.

    Assumes naive timestamps are already Eastern Time.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=NY_TZ)

    return ts.astimezone(NY_TZ)


def unix_to_ny_time(unix_timestamp: int) -> datetime:
    """
    Convert a FinnHub UNIX timestamp to America/New_York time.
    """
    return datetime.fromtimestamp(
        unix_timestamp,
        tz=UTC_TZ
    ).astimezone(NY_TZ)


def date_string(dt: datetime) -> str:
    """
    Convert a datetime to YYYY-MM-DD format for FinnHub.
    """
    return dt.strftime("%Y-%m-%d")


def format_news_item(item):
    """
    Convert a raw FinnHub article into the project news format.

    Returns None if the item does not contain a valid datetime.
    """
    try:
        unix_timestamp = item.get("datetime")

        if unix_timestamp is None:
            return None

        published_at = unix_to_ny_time(unix_timestamp)

    except (TypeError, ValueError, OSError):
        return None

    return {
        "headline": item.get("headline"),
        "summary": item.get("summary"),
        "source": item.get("source"),
        "url": item.get("url"),
        "published_at_et": published_at,
        "published_unix": unix_timestamp,
    }


def filter_news_before_cutoff(news_items, cutoff_timestamp):
    """
    Keep only news published at or before the T0 cutoff timestamp.

    This prevents look-ahead leakage.
    """
    cutoff_timestamp = to_ny_time(cutoff_timestamp)

    filtered = []

    for item in news_items:
        formatted_item = format_news_item(item)

        if formatted_item is None:
            continue

        if formatted_item["published_at_et"] <= cutoff_timestamp:
            filtered.append(formatted_item)

    return filtered


def sort_news_newest_first(news_items):
    """
    Sort news by publish time, newest first.
    """
    return sorted(
        news_items,
        key=lambda item: item["published_at_et"],
        reverse=True
    )


def _finnhub_get(ticker, _from, to, api_key):
    """
    Single network boundary for FinnHub. Tests monkeypatch this.

    Returns the raw company-news list (list of dicts) for the date range
    [_from, to], inclusive, as YYYY-MM-DD strings.
    """
    import finnhub

    client = finnhub.Client(api_key=api_key)

    return client.company_news(ticker, _from=_from, to=to)


def _atomic_write(path, content):
    """
    Write content to path via a temporary file + rename, so partial
    writes from interrupted runs do not corrupt the cache.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    tmp_path.replace(path)


def _news_cache_path(cache_dir, ticker, _from, to):
    """
    Cache file for one (ticker, date-range) request:
    <cache_dir>/<TICKER>/<from>_<to>.json
    """
    return Path(cache_dir) / ticker.upper() / f"{_from}_{to}.json"


def _read_news_cache(cache_dir, ticker, _from, to):
    """
    Return the cached raw news list for this request, or None if absent.
    """
    path = _news_cache_path(cache_dir, ticker, _from, to)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("raw", [])


def _write_news_cache(cache_dir, ticker, _from, to, raw):
    """
    Atomically persist the raw FinnHub response for this request.
    """
    path = _news_cache_path(cache_dir, ticker, _from, to)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "ticker": ticker.upper(),
        "from": _from,
        "to": to,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw": raw,
    }

    _atomic_write(path, json.dumps(payload, indent=2))


def fetch_company_news(
    ticker,
    cutoff_timestamp,
    api_key,
    lookback_days=7,
    cache_dir=None,
):
    """
    Fetch company news from FinnHub and filter it to prevent leakage.

    Inputs:
        ticker:
            Stock ticker, such as AAPL.

        cutoff_timestamp:
            The T0 close timestamp from Task 2.

        api_key:
            FinnHub API key.

        lookback_days:
            Number of days before the cutoff to request.

        cache_dir:
            Optional directory for an on-disk request cache. When set, a
            repeated (ticker, date-range) request is served from disk
            instead of hitting the API (FinnHub's free tier is quota
            limited). Pass DEFAULT_NEWS_CACHE in production; None disables
            caching (used by tests and pure/offline paths). Mirrors the
            EDGAR / RAG cache convention.

    Output:
        List of company news dictionaries published at or before
        the cutoff timestamp.
    """
    cutoff_timestamp = to_ny_time(cutoff_timestamp)

    start_date = cutoff_timestamp - timedelta(days=lookback_days)

    _from = date_string(start_date)
    to = date_string(cutoff_timestamp)

    # The cache stores the RAW API response (what costs quota). Filtering,
    # the exact cutoff, and sorting are cheap and applied on every read, so
    # they can change without invalidating the cache. The raw request is
    # date-granular, matching FinnHub itself; the precise intraday cutoff is
    # still enforced below, so leakage protection is unchanged.
    raw_news = None

    if cache_dir is not None:
        raw_news = _read_news_cache(cache_dir, ticker, _from, to)

    if raw_news is None:
        raw_news = _finnhub_get(ticker, _from, to, api_key)

        if cache_dir is not None:
            _write_news_cache(cache_dir, ticker, _from, to, raw_news)

    filtered_news = filter_news_before_cutoff(
        raw_news,
        cutoff_timestamp
    )

    return sort_news_newest_first(filtered_news)
