from __future__ import annotations

from datetime import timedelta

import pytest

from grizz.utils.interval import (
    find_time_unit,
    interval_to_strftime_format,
    interval_to_timedelta,
    time_unit_to_strftime_format,
)

####################################
#     Tests for find_time_unit     #
####################################


@pytest.mark.parametrize(
    ("interval", "time_unit"),
    [
        # ns
        ("1ns", "ns"),
        ("5ns", "ns"),
        ("3d12h4m1s1ms1us1ns", "ns"),
        # us
        ("1us", "us"),
        ("5us", "us"),
        ("3d12h4m1s1ms1us", "us"),
        # ms
        ("1ms", "ms"),
        ("5ms", "ms"),
        ("3d12h4m1s1ms", "ms"),
        # s
        ("1s", "s"),
        ("5s", "s"),
        ("3d12h4m1s", "s"),
        # m
        ("1m", "m"),
        ("5m", "m"),
        ("3d12h4m", "m"),
        # h
        ("1h", "h"),
        ("5h", "h"),
        ("3d12h", "h"),
        # d
        ("1d", "d"),
        ("5d", "d"),
        ("3w12d", "d"),
        # w
        ("1w", "w"),
        ("5w", "w"),
        ("3mo12w", "w"),
        # mo
        ("1mo", "mo"),
        ("5mo", "mo"),
        ("3y12mo", "mo"),
        # q
        ("1q", "q"),
        ("5q", "q"),
        ("3y12q", "q"),
        # y
        ("1y", "y"),
        ("5y", "y"),
        ("3y12y", "y"),
    ],
)
def test_find_time_unit(interval: str, time_unit: str) -> None:
    assert find_time_unit(interval) == time_unit


@pytest.mark.parametrize(
    "interval",
    ["ns", "us", "ms", "s", "m", "h", "d", "w", "mo", "q", "y", "abc"],
)
def test_find_time_unit_incorrect(interval: str) -> None:
    with pytest.raises(RuntimeError, match=r"could not find the time unit of"):
        find_time_unit(interval)


###############################################
#     Tests for interval_to_strftime_format     #
###############################################


def test_interval_to_strftime_format_minute() -> None:
    assert interval_to_strftime_format("3d12h4m") == "%Y-%m-%d %H:%M"


def test_interval_to_strftime_format_month() -> None:
    assert interval_to_strftime_format("3y1mo") == "%Y-%m"


def test_interval_to_strftime_format_invalid() -> None:
    with pytest.raises(RuntimeError, match=r"could not find the time unit of invalid"):
        interval_to_strftime_format("invalid")


####################################
#     Tests for find_time_unit     #
####################################


@pytest.mark.parametrize(
    ("interval", "time_delta"),
    [
        ("2w", timedelta(days=14)),
        ("12w", timedelta(days=84)),
        ("5d", timedelta(days=5)),
        ("55d", timedelta(days=55)),
        ("6h", timedelta(hours=6)),
        ("16h", timedelta(hours=16)),
        ("7m", timedelta(minutes=7)),
        ("27m", timedelta(minutes=27)),
        ("2s", timedelta(seconds=2)),
        ("42s", timedelta(seconds=42)),
        ("3ms", timedelta(milliseconds=3)),
        ("23ms", timedelta(milliseconds=23)),
        ("7us", timedelta(microseconds=7)),
        ("37us", timedelta(microseconds=37)),
        ("7ns", timedelta(microseconds=0.007)),
        ("37ns", timedelta(microseconds=0.037)),
        ("5d1h2m", timedelta(days=5, hours=1, minutes=2)),
        ("15d18h42m", timedelta(days=15, hours=18, minutes=42)),
        ("", timedelta()),
        ("1mo", timedelta()),
    ],
)
def test_interval_to_timedelta(interval: str, time_delta: timedelta) -> None:
    assert interval_to_timedelta(interval) == time_delta


##################################################
#     Tests for time_unit_to_strftime_format     #
##################################################


def test_time_unit_to_strftime_format_minute() -> None:
    assert time_unit_to_strftime_format("m") == "%Y-%m-%d %H:%M"


def test_time_unit_to_strftime_format_month() -> None:
    assert time_unit_to_strftime_format("mo") == "%Y-%m"


def test_time_unit_to_strftime_format_invalid() -> None:
    with pytest.raises(
        RuntimeError, match=r"Incorrect time unit invalid. The valid time units are:"
    ):
        time_unit_to_strftime_format("invalid")
