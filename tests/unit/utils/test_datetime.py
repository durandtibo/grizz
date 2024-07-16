from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from grizz.utils.datetime import find_end_datetime, to_datetime

#######################################
#     Tests for find_end_datetime     #
#######################################


@pytest.mark.parametrize(
    ("interval", "end_datetime"),
    [
        (
            timedelta(days=1),
            datetime(year=2020, month=6, day=23, hour=4, minute=0, second=0, tzinfo=timezone.utc),
        ),
        (
            timedelta(hours=1),
            datetime(year=2020, month=5, day=13, hour=22, minute=0, second=0, tzinfo=timezone.utc),
        ),
        (
            timedelta(minutes=1),
            datetime(year=2020, month=5, day=12, hour=4, minute=42, second=0, tzinfo=timezone.utc),
        ),
        (
            "1d",
            datetime(year=2020, month=6, day=23, hour=4, minute=0, second=0, tzinfo=timezone.utc),
        ),
        (
            "1h",
            datetime(year=2020, month=5, day=13, hour=22, minute=0, second=0, tzinfo=timezone.utc),
        ),
        (
            "1m",
            datetime(year=2020, month=5, day=12, hour=4, minute=42, second=0, tzinfo=timezone.utc),
        ),
        (
            "1s",
            datetime(year=2020, month=5, day=12, hour=4, minute=0, second=42, tzinfo=timezone.utc),
        ),
    ],
)
def test_find_end_datetime_interval(interval: str | timedelta, end_datetime: datetime) -> None:
    assert (
        find_end_datetime(
            start=datetime(
                year=2020, month=5, day=12, hour=4, minute=0, second=0, tzinfo=timezone.utc
            ),
            interval=interval,
            periods=42,
        )
        == end_datetime
    )


@pytest.mark.parametrize(
    ("periods", "end_datetime"),
    [
        (0, datetime(year=2020, month=5, day=12, hour=4, minute=0, second=0, tzinfo=timezone.utc)),
        (1, datetime(year=2020, month=5, day=12, hour=5, minute=0, second=0, tzinfo=timezone.utc)),
        (4, datetime(year=2020, month=5, day=12, hour=8, minute=0, second=0, tzinfo=timezone.utc)),
        (
            12,
            datetime(year=2020, month=5, day=12, hour=16, minute=0, second=0, tzinfo=timezone.utc),
        ),
    ],
)
def test_find_end_datetime_periods(periods: int, end_datetime: datetime) -> None:
    assert (
        find_end_datetime(
            start=datetime(
                year=2020, month=5, day=12, hour=4, minute=0, second=0, tzinfo=timezone.utc
            ),
            interval="1h",
            periods=periods,
        )
        == end_datetime
    )


@pytest.mark.parametrize(
    ("start_datetime", "end_datetime"),
    [
        (
            datetime(year=2020, month=5, day=12, hour=4, minute=0, second=0, tzinfo=timezone.utc),
            datetime(year=2020, month=5, day=12, hour=8, minute=0, second=0, tzinfo=timezone.utc),
        ),
        (
            datetime(
                year=2020, month=5, day=12, hour=4, minute=0, second=0, tzinfo=timezone(timedelta())
            ),
            datetime(
                year=2020, month=5, day=12, hour=8, minute=0, second=0, tzinfo=timezone(timedelta())
            ),
        ),
        (
            datetime(
                year=2020,
                month=5,
                day=12,
                hour=4,
                minute=0,
                second=0,
                tzinfo=timezone(timedelta(hours=4)),
            ),
            datetime(
                year=2020,
                month=5,
                day=12,
                hour=8,
                minute=0,
                second=0,
                tzinfo=timezone(timedelta(hours=4)),
            ),
        ),
        (
            date(year=2020, month=5, day=12),
            datetime(year=2020, month=5, day=12, hour=4, minute=0, second=0, tzinfo=timezone.utc),
        ),
    ],
)
def test_find_end_datetime_start(start_datetime: datetime | date, end_datetime: datetime) -> None:
    assert (
        find_end_datetime(
            start=start_datetime,
            interval="1h",
            periods=4,
        )
        == end_datetime
    )


#################################
#     Tests for to_datetime     #
#################################


def test_to_datetime_date() -> None:
    assert to_datetime(date(year=2020, month=5, day=12)) == datetime(
        year=2020, month=5, day=12, hour=0, minute=0, second=0, tzinfo=timezone.utc
    )


def test_to_datetime_datetime() -> None:
    assert to_datetime(
        datetime(year=2020, month=5, day=12, hour=4, tzinfo=timezone.utc)
    ) == datetime(year=2020, month=5, day=12, hour=4, minute=0, second=0, tzinfo=timezone.utc)
