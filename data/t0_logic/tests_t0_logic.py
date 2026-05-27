from datetime import datetime
from zoneinfo import ZoneInfo

from t0_logic import compute_t0


NY = ZoneInfo("America/New_York")


def test_premarket_filing_same_day():
    """
    Filing before market open on a trading day:
    T0 should be the same day.
    """
    result = compute_t0(
        datetime(2026, 2, 3, 8, 0, tzinfo=NY)
    )

    assert str(result["t0_date"]) == "2026-02-03"


def test_intraday_filing_next_trading_day():
    """
    Intraday filing:
    T0 should roll to the next trading day.
    """
    result = compute_t0(
        datetime(2026, 2, 3, 12, 0, tzinfo=NY)
    )

    assert str(result["t0_date"]) == "2026-02-04"


def test_after_hours_filing_next_trading_day():
    """
    After-hours filing:
    T0 should roll to the next trading day.
    """
    result = compute_t0(
        datetime(2026, 2, 3, 17, 0, tzinfo=NY)
    )

    assert str(result["t0_date"]) == "2026-02-04"


def test_weekend_filing_next_trading_day():
    """
    Weekend filing:
    T0 should roll to the next trading day.
    """
    result = compute_t0(
        datetime(2026, 2, 7, 12, 0, tzinfo=NY)  # Saturday
    )

    assert str(result["t0_date"]) == "2026-02-09"


def test_holiday_filing_next_trading_day():
    """
    Holiday filing:
    T0 should roll to the next trading day.

    Example:
    New Year's Day 2026 = NYSE holiday
    """
    result = compute_t0(
        datetime(2026, 1, 1, 12, 0, tzinfo=NY)
    )

    assert str(result["t0_date"]) == "2026-01-02"


def test_exact_market_open_rolls_forward():
    """
    Filing exactly at 9:30 AM ET is NOT before open,
    so it rolls to the next trading day.
    """
    result = compute_t0(
        datetime(2026, 2, 3, 9, 30, tzinfo=NY)
    )

    assert str(result["t0_date"]) == "2026-02-04"


def test_target_date_is_fifth_trading_day_after_t0():
    """
    Verify correct computation of the
    5th trading day after T0.
    """
    result = compute_t0(
        datetime(2026, 2, 3, 8, 0, tzinfo=NY)
    )

    assert str(result["t0_date"]) == "2026-02-03"

    # Trading days after Feb 3:
    # Feb 4, 5, 6, 9, 10
    assert str(result["target_date"]) == "2026-02-10"

def test_early_close_cutoff_timestamp():
    """
    Early-close trading day:
    cutoff timestamp should use the NYSE calendar close,
    not assume a fixed 4:00 PM close.

    Example:
    Black Friday 2026 is an early-close session
    in the NYSE calendar used by this test.
    """
    result = compute_t0(
        datetime(2026, 11, 27, 8, 0, tzinfo=NY)
    )

    assert str(result["t0_date"]) == "2026-11-27"
    assert result["cutoff_timestamp_et"].hour == 13
    assert result["cutoff_timestamp_et"].minute == 0
    assert str(result["cutoff_timestamp_et"].tzinfo) == "America/New_York"
