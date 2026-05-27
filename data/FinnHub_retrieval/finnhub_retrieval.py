from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

#import finnhub


NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


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


def fetch_company_news(
    ticker,
    cutoff_timestamp,
    api_key,
    lookback_days=7,
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

    Output:
        List of company news dictionaries published at or before
        the cutoff timestamp.
    """
    import finnhub    

    cutoff_timestamp = to_ny_time(cutoff_timestamp)

    start_date = cutoff_timestamp - timedelta(days=lookback_days)

    client = finnhub.Client(api_key=api_key)

    raw_news = client.company_news(
        ticker,
        _from=date_string(start_date),
        to=date_string(cutoff_timestamp),
    )

    filtered_news = filter_news_before_cutoff(
        raw_news,
        cutoff_timestamp
    )

    return sort_news_newest_first(filtered_news)
