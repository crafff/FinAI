import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


# Where the on-disk price cache lives by default. Callers pass
# cache_dir=DEFAULT_PRICES_CACHE to turn caching on; tests pass tmp_path.
DEFAULT_PRICES_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "prices"


def get_close_on_date(close_series, target_date):
    """
    Return the closing price for a specific trading date.

    close_series is a mapping of datetime.date -> close (float), as
    produced by _download_close_series. T0 and the target date are both
    trading days by construction (see t0_logic), so a missing key means
    the price data genuinely does not cover that date and we raise rather
    than silently returning a neighbour's price.
    """
    if target_date not in close_series:
        raise LookupError(
            f"No closing price for {target_date} in the downloaded series."
        )

    return close_series[target_date]


def get_price_trend(close_series, t0_date, trend_days=30):
    """
    Return the pre-release price trend: the closes for the last
    `trend_days` trading days up to and including T0.

    Including T0 is intentional - T0's close is the baseline and the
    information cutoff, so the trend ends exactly at the anchor with no
    look-ahead. Dates strictly after T0 are excluded.

    Returns a list of {"date": date, "close": float} sorted oldest-first.
    """
    eligible_dates = sorted(d for d in close_series if d <= t0_date)

    selected = eligible_dates[-trend_days:]

    return [{"date": d, "close": close_series[d]} for d in selected]


def _download_close_series(ticker, start_date, end_date):
    """
    Single network boundary for price data. Tests monkeypatch this.

    Downloads daily OHLC from yfinance for [start_date, end_date] and
    returns a mapping of datetime.date -> raw close (float).

    auto_adjust is disabled so the returned close is the actual quoted
    close that downstream evaluation compares against, not a
    split/dividend-adjusted series.
    """
    import yfinance as yf

    # yfinance treats `end` as exclusive, so add a day to include it.
    frame = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )

    if frame is None or frame.empty:
        return {}

    close = frame["Close"]

    # Multi-ticker downloads return a column per ticker; collapse to one.
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    series = {}

    for idx, value in close.items():
        ts = idx.date() if hasattr(idx, "date") else idx

        if value is None:
            continue

        try:
            series[ts] = float(value)
        except (TypeError, ValueError):
            continue

    return series


def _atomic_write(path, content):
    """
    Write content to path via a temporary file + rename, so partial writes
    from interrupted runs do not corrupt the cache.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    tmp_path.replace(path)


def _prices_cache_path(cache_dir, ticker, t0_date, target_date, trend_days):
    """
    Cache file for one price request:
    <cache_dir>/<TICKER>/<t0>_<target>_t<trend_days>.json
    """
    name = f"{t0_date.isoformat()}_{target_date.isoformat()}_t{trend_days}.json"
    return Path(cache_dir) / ticker.upper() / name


def _read_prices_cache(cache_dir, ticker, t0_date, target_date, trend_days):
    """
    Return the cached close series (date -> float) for this request, or None
    if absent. Date keys are stored as ISO strings and rehydrated here.
    """
    path = _prices_cache_path(cache_dir, ticker, t0_date, target_date, trend_days)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload.get("raw", {})

    return {date.fromisoformat(k): float(v) for k, v in raw.items()}


def _write_prices_cache(
    cache_dir, ticker, t0_date, target_date, trend_days, close_series
):
    """
    Atomically persist the raw downloaded close series for this request.
    The close series is the unit that costs a yfinance download; the
    baseline / target / trend are cheap to re-derive on read.
    """
    path = _prices_cache_path(cache_dir, ticker, t0_date, target_date, trend_days)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "ticker": ticker.upper(),
        "t0_date": t0_date.isoformat(),
        "target_date": target_date.isoformat(),
        "trend_days": trend_days,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw": {d.isoformat(): v for d, v in close_series.items()},
    }

    _atomic_write(path, json.dumps(payload, indent=2))


def fetch_prices(ticker, t0_date, target_date, trend_days=30, cache_dir=None):
    """
    Fetch the baseline price, target price, and pre-release trend for one
    company around its filing.

    Inputs:
        ticker:        stock ticker, e.g. "AAPL".
        t0_date:       T0 date from t0_logic.compute_t0.
        target_date:   5th-trading-day date from t0_logic.compute_t0.
        trend_days:    how many trailing trading days of trend to return.
        cache_dir:     optional directory for an on-disk cache. When set, a
                       repeated (ticker, t0, target, trend_days) request is
                       served from disk instead of re-downloading from
                       yfinance. Pass DEFAULT_PRICES_CACHE in production;
                       None disables caching (tests / offline). Mirrors the
                       EDGAR / FinnHub cache convention.

    Output dict:
        ticker
        t0_date
        target_date
        baseline_price        close on T0
        target_price          close on the target date
        pre_release_trend     list of {date, close}, oldest-first,
                              ending at (and including) T0

    The single download spans enough calendar days before T0 to cover
    `trend_days` trading days, and through the target date.
    """
    # Calendar buffer: ~7 calendar days per 5 trading days, plus slack
    # for holidays, so we always capture `trend_days` trading days.
    lookback_calendar_days = trend_days * 2 + 10
    start_date = t0_date - timedelta(days=lookback_calendar_days)

    # The cache stores the RAW downloaded close series (what costs the
    # network round-trip). The baseline/target/trend are re-derived on every
    # read. An empty result is never cached, so a transient failure is not
    # persisted.
    close_series = None

    if cache_dir is not None:
        close_series = _read_prices_cache(
            cache_dir, ticker, t0_date, target_date, trend_days
        )

    if not close_series:
        close_series = _download_close_series(ticker, start_date, target_date)

        if cache_dir is not None and close_series:
            _write_prices_cache(
                cache_dir, ticker, t0_date, target_date, trend_days, close_series
            )

    baseline_price = get_close_on_date(close_series, t0_date)
    target_price = get_close_on_date(close_series, target_date)
    pre_release_trend = get_price_trend(close_series, t0_date, trend_days)

    return {
        "ticker": ticker.upper(),
        "t0_date": t0_date,
        "target_date": target_date,
        "baseline_price": baseline_price,
        "target_price": target_price,
        "pre_release_trend": pre_release_trend,
    }
