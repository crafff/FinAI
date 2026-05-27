from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal


NY_TZ = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)

nyse = mcal.get_calendar("NYSE")


def to_ny_time(ts: datetime) -> datetime:
    """
    Normalize timestamps to America/New_York.

    Assumes naive timestamps are already Eastern Time.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=NY_TZ)

    return ts.astimezone(NY_TZ)


def get_trading_days(start_date, end_date):
    """
    Return a list of NYSE trading dates between
    start_date and end_date.
    """
    schedule = nyse.schedule(
        start_date=start_date,
        end_date=end_date
    )

    return list(schedule.index.date)


def is_trading_day(date):
    """
    Check whether a date is a valid NYSE trading day.
    """
    return date in get_trading_days(date, date)


def next_trading_day(after_date):
    """
    Return the next NYSE trading day after after_date.
    """
    end_date = pd.Timestamp(after_date) + pd.Timedelta(days=14)

    trading_days = get_trading_days(
        after_date,
        end_date.date()
    )

    for day in trading_days:
        if day > after_date:
            return day

    raise ValueError("No next trading day found.")


def fifth_trading_day_after(t0_date):
    """
    Return the 5th trading day after T0.
    """
    end_date = pd.Timestamp(t0_date) + pd.Timedelta(days=20)

    trading_days = get_trading_days(
        t0_date,
        end_date.date()
    )

    future_days = [
        day for day in trading_days
        if day > t0_date
    ]

    if len(future_days) < 5:
        raise ValueError(
            "Could not locate 5th trading day after T0."
        )

    return future_days[4]


def market_close_timestamp(trading_date):
    """
    Return the official NYSE market close timestamp
    for a trading day.

    Uses the exchange calendar directly so the logic
    remains correct on early-close trading sessions.
    """
    schedule = nyse.schedule(
        start_date=trading_date,
        end_date=trading_date
    )

    if schedule.empty:
        raise ValueError(
            f"{trading_date} is not a trading day."
        )

    close_ts = schedule.iloc[0]["market_close"]

    return close_ts.to_pydatetime().astimezone(NY_TZ)


def compute_t0(filing_timestamp: datetime):
    """
    Compute:
        - T0 date
        - information cutoff timestamp
        - target evaluation date

    Unified rule:
        pre-open filing on trading day -> same day
        otherwise -> next trading day
    """

    filing_timestamp = to_ny_time(filing_timestamp)

    filing_date = filing_timestamp.date()
    filing_time = filing_timestamp.time()

    if (
        is_trading_day(filing_date)
        and filing_time < MARKET_OPEN
    ):
        t0_date = filing_date

    else:
        t0_date = next_trading_day(filing_date)

    cutoff_timestamp = market_close_timestamp(t0_date)

    target_date = fifth_trading_day_after(t0_date)

    return {
        "filing_timestamp_et": filing_timestamp,
        "t0_date": t0_date,
        "cutoff_timestamp_et": cutoff_timestamp,
        "target_date": target_date,
    }
