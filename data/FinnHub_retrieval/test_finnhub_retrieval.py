from datetime import datetime
from zoneinfo import ZoneInfo

import finnhub_retrieval
from finnhub_retrieval import (
    to_ny_time,
    unix_to_ny_time,
    filter_news_before_cutoff,
    sort_news_newest_first,
    fetch_company_news,
    _news_cache_path,
)


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def make_unix_timestamp(year, month, day, hour, minute, tz=NY):
    """
    Helper for creating mock FinnHub UNIX timestamps.
    """
    return int(
        datetime(
            year,
            month,
            day,
            hour,
            minute,
            tzinfo=tz
        ).timestamp()
    )


def test_to_ny_time_converts_utc_to_eastern():
    ts = datetime(2026, 2, 3, 21, 0, tzinfo=UTC)

    result = to_ny_time(ts)

    assert result.hour == 16
    assert result.tzinfo.key == "America/New_York"


def test_to_ny_time_treats_naive_as_eastern():
    ts = datetime(2026, 2, 3, 16, 0)

    result = to_ny_time(ts)

    assert result.hour == 16
    assert result.tzinfo.key == "America/New_York"


def test_unix_to_ny_time():
    unix_ts = make_unix_timestamp(2026, 2, 3, 15, 30)

    result = unix_to_ny_time(unix_ts)

    assert result.hour == 15
    assert result.minute == 30
    assert result.tzinfo.key == "America/New_York"


def test_filter_news_before_cutoff_keeps_before_and_at_cutoff():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_news = [
        {
            "headline": "Before cutoff",
            "summary": "Valid article",
            "source": "Test",
            "url": "https://example.com/before",
            "datetime": make_unix_timestamp(2026, 2, 3, 15, 30),
        },
        {
            "headline": "At cutoff",
            "summary": "Still valid",
            "source": "Test",
            "url": "https://example.com/at",
            "datetime": make_unix_timestamp(2026, 2, 3, 16, 0),
        },
    ]

    result = filter_news_before_cutoff(raw_news, cutoff)

    assert len(result) == 2
    assert result[0]["headline"] == "Before cutoff"
    assert result[1]["headline"] == "At cutoff"


def test_filter_news_before_cutoff_removes_after_cutoff():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_news = [
        {
            "headline": "After cutoff",
            "summary": "Leaky article",
            "source": "Test",
            "url": "https://example.com/after",
            "datetime": make_unix_timestamp(2026, 2, 3, 16, 1),
        },
    ]

    result = filter_news_before_cutoff(raw_news, cutoff)

    assert result == []


def test_filter_news_handles_naive_cutoff_as_eastern():
    cutoff = datetime(2026, 2, 3, 16, 0)

    raw_news = [
        {
            "headline": "At cutoff",
            "summary": "Valid article",
            "source": "Test",
            "url": "https://example.com/at",
            "datetime": make_unix_timestamp(2026, 2, 3, 16, 0),
        }
    ]

    result = filter_news_before_cutoff(raw_news, cutoff)

    assert len(result) == 1
    assert result[0]["headline"] == "At cutoff"


def test_filter_news_skips_items_without_datetime():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_news = [
        {
            "headline": "Missing timestamp",
            "summary": "Malformed article",
            "source": "Test",
            "url": "https://example.com/missing",
        },
        {
            "headline": "Valid article",
            "summary": "Has timestamp",
            "source": "Test",
            "url": "https://example.com/valid",
            "datetime": make_unix_timestamp(2026, 2, 3, 15, 30),
        },
    ]

    result = filter_news_before_cutoff(raw_news, cutoff)

    assert len(result) == 1
    assert result[0]["headline"] == "Valid article"


def test_filter_news_skips_items_with_invalid_datetime():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_news = [
        {
            "headline": "Invalid timestamp",
            "summary": "Malformed article",
            "source": "Test",
            "url": "https://example.com/invalid",
            "datetime": "not-a-timestamp",
        },
        {
            "headline": "Valid article",
            "summary": "Has timestamp",
            "source": "Test",
            "url": "https://example.com/valid",
            "datetime": make_unix_timestamp(2026, 2, 3, 15, 30),
        },
    ]

    result = filter_news_before_cutoff(raw_news, cutoff)

    assert len(result) == 1
    assert result[0]["headline"] == "Valid article"


def test_sort_news_newest_first():
    older = {
        "headline": "Older",
        "published_at_et": datetime(2026, 2, 3, 10, 0, tzinfo=NY),
    }

    newer = {
        "headline": "Newer",
        "published_at_et": datetime(2026, 2, 3, 15, 0, tzinfo=NY),
    }

    result = sort_news_newest_first([older, newer])

    assert result[0]["headline"] == "Newer"
    assert result[1]["headline"] == "Older"

def test_filter_then_sort_news_newest_first():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_news = [
        {
            "headline": "Older valid article",
            "datetime": make_unix_timestamp(2026, 2, 3, 10, 0),
        },
        {
            "headline": "After cutoff article",
            "datetime": make_unix_timestamp(2026, 2, 3, 16, 1),
        },
        {
            "headline": "Newer valid article",
            "datetime": make_unix_timestamp(2026, 2, 3, 15, 0),
        },
    ]

    filtered = filter_news_before_cutoff(raw_news, cutoff)
    result = sort_news_newest_first(filtered)

    assert len(result) == 2
    assert result[0]["headline"] == "Newer valid article"
    assert result[1]["headline"] == "Older valid article"


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------

def _raw_item(headline, year, month, day, hour, minute):
    return {
        "headline": headline,
        "summary": "s",
        "source": "Test",
        "url": "https://example.com/" + headline,
        "datetime": make_unix_timestamp(year, month, day, hour, minute),
    }


def _install_fake_get(monkeypatch, raw, call_log):
    def fake_get(ticker, _from, to, api_key):
        call_log.append((ticker, _from, to))
        return raw

    monkeypatch.setattr(finnhub_retrieval, "_finnhub_get", fake_get)


def test_fetch_company_news_caches_and_reuses(monkeypatch, tmp_path):
    cutoff = datetime(2025, 1, 8, 16, 0, tzinfo=NY)
    raw = [_raw_item("Cached", 2025, 1, 7, 10, 0)]

    call_log = []
    _install_fake_get(monkeypatch, raw, call_log)

    first = fetch_company_news("AAPL", cutoff, "key", cache_dir=tmp_path)
    second = fetch_company_news("AAPL", cutoff, "key", cache_dir=tmp_path)

    # The second identical request is served from disk: one API hit total.
    assert len(call_log) == 1
    assert [i["headline"] for i in first] == ["Cached"]
    assert [i["headline"] for i in second] == ["Cached"]


def test_fetch_company_news_writes_cache_file(monkeypatch, tmp_path):
    cutoff = datetime(2025, 1, 8, 16, 0, tzinfo=NY)
    _install_fake_get(monkeypatch, [_raw_item("X", 2025, 1, 7, 10, 0)], [])

    fetch_company_news("aapl", cutoff, "key", cache_dir=tmp_path)

    expected = _news_cache_path(tmp_path, "AAPL", "2025-01-01", "2025-01-08")
    assert expected.exists()


def test_fetch_company_news_no_cache_dir_always_calls_api(monkeypatch):
    cutoff = datetime(2025, 1, 8, 16, 0, tzinfo=NY)
    call_log = []
    _install_fake_get(monkeypatch, [_raw_item("Y", 2025, 1, 7, 10, 0)], call_log)

    fetch_company_news("AAPL", cutoff, "key")
    fetch_company_news("AAPL", cutoff, "key")

    # No cache_dir -> every call hits the network.
    assert len(call_log) == 2
