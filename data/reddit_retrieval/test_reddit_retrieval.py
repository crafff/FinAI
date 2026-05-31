from datetime import datetime
from zoneinfo import ZoneInfo

import reddit_retrieval
from reddit_retrieval import (
    to_ny_time,
    unix_to_ny_time,
    format_post,
    filter_posts_before_cutoff,
    sort_posts_newest_first,
    fetch_reddit_posts,
    _posts_cache_path,
)


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def make_unix_timestamp(year, month, day, hour, minute, tz=NY):
    """
    Helper for creating mock Reddit created_utc UNIX timestamps.
    """
    return int(
        datetime(year, month, day, hour, minute, tzinfo=tz).timestamp()
    )


class FakeSubmission:
    """
    Minimal stand-in for a PRAW submission object, to verify that
    format_post reads attributes as well as dicts.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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


def test_format_post_reads_praw_object_attributes():
    submission = FakeSubmission(
        id="abc123",
        title="AAPL to the moon",
        selftext="long thesis",
        subreddit="wallstreetbets",
        score=42,
        num_comments=7,
        url="https://reddit.com/r/wallstreetbets/abc123",
        permalink="/r/wallstreetbets/abc123",
        created_utc=make_unix_timestamp(2026, 2, 3, 15, 30),
    )

    result = format_post(submission)

    assert result["id"] == "abc123"
    assert result["title"] == "AAPL to the moon"
    assert result["body"] == "long thesis"
    assert result["subreddit"] == "wallstreetbets"
    assert result["score"] == 42
    assert result["published_at_et"].hour == 15


def test_format_post_returns_none_without_created_utc():
    assert format_post({"title": "no timestamp"}) is None


def test_format_post_returns_none_on_invalid_created_utc():
    assert format_post({"title": "bad", "created_utc": "not-a-number"}) is None


def test_filter_posts_keeps_before_and_at_cutoff():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_posts = [
        {"title": "Before cutoff",
         "created_utc": make_unix_timestamp(2026, 2, 3, 15, 30)},
        {"title": "At cutoff",
         "created_utc": make_unix_timestamp(2026, 2, 3, 16, 0)},
    ]

    result = filter_posts_before_cutoff(raw_posts, cutoff)

    assert len(result) == 2
    titles = {post["title"] for post in result}
    assert titles == {"Before cutoff", "At cutoff"}


def test_filter_posts_removes_after_cutoff():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_posts = [
        {"title": "After cutoff",
         "created_utc": make_unix_timestamp(2026, 2, 3, 16, 1)},
    ]

    assert filter_posts_before_cutoff(raw_posts, cutoff) == []


def test_filter_posts_handles_naive_cutoff_as_eastern():
    cutoff = datetime(2026, 2, 3, 16, 0)

    raw_posts = [
        {"title": "At cutoff",
         "created_utc": make_unix_timestamp(2026, 2, 3, 16, 0)},
    ]

    result = filter_posts_before_cutoff(raw_posts, cutoff)

    assert len(result) == 1
    assert result[0]["title"] == "At cutoff"


def test_filter_posts_skips_items_without_created_utc():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_posts = [
        {"title": "Missing timestamp"},
        {"title": "Valid",
         "created_utc": make_unix_timestamp(2026, 2, 3, 15, 30)},
    ]

    result = filter_posts_before_cutoff(raw_posts, cutoff)

    assert len(result) == 1
    assert result[0]["title"] == "Valid"


def test_sort_posts_newest_first():
    older = {"title": "Older",
             "published_at_et": datetime(2026, 2, 3, 10, 0, tzinfo=NY)}
    newer = {"title": "Newer",
             "published_at_et": datetime(2026, 2, 3, 15, 0, tzinfo=NY)}

    result = sort_posts_newest_first([older, newer])

    assert result[0]["title"] == "Newer"
    assert result[1]["title"] == "Older"


def test_filter_then_sort_posts():
    cutoff = datetime(2026, 2, 3, 16, 0, tzinfo=NY)

    raw_posts = [
        {"title": "Older valid",
         "created_utc": make_unix_timestamp(2026, 2, 3, 10, 0)},
        {"title": "After cutoff",
         "created_utc": make_unix_timestamp(2026, 2, 3, 16, 1)},
        {"title": "Newer valid",
         "created_utc": make_unix_timestamp(2026, 2, 3, 15, 0)},
    ]

    filtered = filter_posts_before_cutoff(raw_posts, cutoff)
    result = sort_posts_newest_first(filtered)

    assert len(result) == 2
    assert result[0]["title"] == "Newer valid"
    assert result[1]["title"] == "Older valid"


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------

def _raw_post(post_id, created_utc):
    return {
        "id": post_id,
        "title": post_id,
        "selftext": "body",
        "subreddit": "stocks",
        "score": 1,
        "num_comments": 0,
        "url": "http://x",
        "permalink": "/r/stocks/" + post_id,
        "created_utc": created_utc,
    }


def _install_fake_search(monkeypatch, raw, call_log):
    def fake_search(query, subreddits, limit, client_id, client_secret, user_agent):
        call_log.append(query)
        return raw

    monkeypatch.setattr(reddit_retrieval, "_reddit_search", fake_search)


def test_fetch_reddit_posts_caches_and_reuses(monkeypatch, tmp_path):
    cutoff = datetime(2025, 1, 2, 16, 0, tzinfo=NY)
    raw = [_raw_post("p1", make_unix_timestamp(2025, 1, 2, 12, 0))]

    call_log = []
    _install_fake_search(monkeypatch, raw, call_log)

    first = fetch_reddit_posts(
        "AAPL", cutoff, client_id="id", client_secret="s", user_agent="ua",
        subreddits=("stocks",), cache_dir=tmp_path,
    )
    second = fetch_reddit_posts(
        "AAPL", cutoff, client_id="id", client_secret="s", user_agent="ua",
        subreddits=("stocks",), cache_dir=tmp_path,
    )

    # Second identical search is served from disk: one API search total.
    assert len(call_log) == 1
    assert [p["id"] for p in first] == ["p1"]
    assert [p["id"] for p in second] == ["p1"]


def test_fetch_reddit_posts_writes_cache_file(monkeypatch, tmp_path):
    cutoff = datetime(2025, 1, 2, 16, 0, tzinfo=NY)
    raw = [_raw_post("p1", make_unix_timestamp(2025, 1, 2, 12, 0))]
    _install_fake_search(monkeypatch, raw, [])

    fetch_reddit_posts(
        "aapl", cutoff, client_id="id", client_secret="s", user_agent="ua",
        subreddits=("stocks",), cache_dir=tmp_path,
    )

    # query defaults to the ticker symbol ("aapl").
    expected = _posts_cache_path(tmp_path, "AAPL", "aapl", ("stocks",), 50)
    assert expected.exists()


def test_fetch_reddit_posts_no_cache_dir_always_calls_api(monkeypatch):
    cutoff = datetime(2025, 1, 2, 16, 0, tzinfo=NY)
    raw = [_raw_post("p1", make_unix_timestamp(2025, 1, 2, 12, 0))]

    call_log = []
    _install_fake_search(monkeypatch, raw, call_log)

    fetch_reddit_posts(
        "AAPL", cutoff, client_id="id", client_secret="s", user_agent="ua",
        subreddits=("stocks",),
    )
    fetch_reddit_posts(
        "AAPL", cutoff, client_id="id", client_secret="s", user_agent="ua",
        subreddits=("stocks",),
    )

    assert len(call_log) == 2
