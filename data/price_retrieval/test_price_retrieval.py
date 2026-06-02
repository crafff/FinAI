from datetime import date

import pytest

import price_retrieval
from price_retrieval import (
    get_close_on_date,
    get_price_trend,
    fetch_prices,
)


def make_series(start, n, start_price=100.0, step=1.0):
    """
    Build a contiguous daily close series of length n starting at `start`.

    Weekends are not modelled; this is a simple deterministic fixture for
    the pure date-selection logic, which does not care about the calendar.
    """
    from datetime import timedelta

    return {
        start + timedelta(days=i): start_price + i * step
        for i in range(n)
    }


def test_get_close_on_date_returns_value():
    series = {date(2026, 2, 3): 150.0, date(2026, 2, 4): 151.0}

    assert get_close_on_date(series, date(2026, 2, 3)) == 150.0


def test_get_close_on_date_missing_raises():
    series = {date(2026, 2, 3): 150.0}

    with pytest.raises(LookupError):
        get_close_on_date(series, date(2026, 2, 4))


def test_get_price_trend_includes_t0_and_excludes_future():
    series = make_series(date(2026, 1, 1), 40)
    t0 = date(2026, 1, 20)

    trend = get_price_trend(series, t0, trend_days=5)

    assert len(trend) == 5
    # Oldest-first, ending exactly at T0.
    assert trend[-1]["date"] == t0
    assert all(item["date"] <= t0 for item in trend)


def test_get_price_trend_oldest_first_ordering():
    series = make_series(date(2026, 1, 1), 10)
    t0 = date(2026, 1, 10)

    trend = get_price_trend(series, t0, trend_days=3)

    dates = [item["date"] for item in trend]

    assert dates == sorted(dates)


def test_get_price_trend_handles_short_history():
    series = make_series(date(2026, 1, 1), 3)
    t0 = date(2026, 1, 3)

    trend = get_price_trend(series, t0, trend_days=30)

    # Fewer than trend_days available -> return what exists, no error.
    assert len(trend) == 3


def test_fetch_prices_integration(monkeypatch):
    t0 = date(2026, 2, 3)
    target = date(2026, 2, 10)

    series = make_series(date(2026, 1, 1), 50)

    captured = {}

    def fake_download(ticker, start_date, end_date):
        captured["ticker"] = ticker
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return series

    monkeypatch.setattr(price_retrieval, "_download_close_series", fake_download)

    result = fetch_prices("aapl", t0, target, trend_days=10)

    assert result["ticker"] == "AAPL"
    assert result["t0_date"] == t0
    assert result["target_date"] == target
    assert result["baseline_price"] == series[t0]
    assert result["target_price"] == series[target]
    assert len(result["pre_release_trend"]) == 10
    assert result["pre_release_trend"][-1]["date"] == t0

    # Download window starts well before T0 and the request is lower-cased
    # ticker passed through unchanged to the boundary.
    assert captured["ticker"] == "aapl"
    assert captured["start_date"] < t0
    assert captured["end_date"] == target


def test_fetch_prices_cache_serves_second_call_from_disk(tmp_path, monkeypatch):
    t0 = date(2026, 2, 3)
    target = date(2026, 2, 10)
    series = make_series(date(2026, 1, 1), 50)

    calls = {"n": 0}

    def fake_download(ticker, start_date, end_date):
        calls["n"] += 1
        return series

    monkeypatch.setattr(price_retrieval, "_download_close_series", fake_download)

    first = fetch_prices("AAPL", t0, target, trend_days=10, cache_dir=tmp_path)
    assert calls["n"] == 1

    # Second identical call is served from disk: no further download, equal result.
    second = fetch_prices("AAPL", t0, target, trend_days=10, cache_dir=tmp_path)
    assert calls["n"] == 1
    assert second["baseline_price"] == first["baseline_price"]
    assert second["target_price"] == first["target_price"]
    assert [p["date"] for p in second["pre_release_trend"]] == \
        [p["date"] for p in first["pre_release_trend"]]

    cache_file = tmp_path / "AAPL" / f"{t0.isoformat()}_{target.isoformat()}_t10.json"
    assert cache_file.exists()


def test_fetch_prices_no_cache_dir_downloads_every_call(monkeypatch):
    t0 = date(2026, 2, 3)
    target = date(2026, 2, 10)
    series = make_series(date(2026, 1, 1), 50)

    calls = {"n": 0}

    def fake_download(ticker, start_date, end_date):
        calls["n"] += 1
        return series

    monkeypatch.setattr(price_retrieval, "_download_close_series", fake_download)

    fetch_prices("AAPL", t0, target, trend_days=10)
    fetch_prices("AAPL", t0, target, trend_days=10)

    assert calls["n"] == 2


def test_fetch_prices_missing_target_raises(monkeypatch):
    t0 = date(2026, 2, 3)
    target = date(2026, 2, 10)

    # Series that covers T0 but not the target date.
    series = make_series(date(2026, 1, 1), 34)  # ends 2026-02-03

    monkeypatch.setattr(
        price_retrieval, "_download_close_series", lambda *a: series
    )

    with pytest.raises(LookupError):
        fetch_prices("AAPL", t0, target)
