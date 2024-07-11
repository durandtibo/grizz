from __future__ import annotations

import pytest

from grizz.utils.period import (
    find_time_unit,
    period_to_strftime_format,
    time_unit_to_strftime_format,
)

####################################
#     Tests for find_time_unit     #
####################################


@pytest.mark.parametrize(
    ("period", "time_unit"),
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
def test_find_time_unit(period: str, time_unit: str) -> None:
    assert find_time_unit(period) == time_unit


@pytest.mark.parametrize(
    "period",
    ["ns", "us", "ms", "s", "m", "h", "d", "w", "mo", "q", "y", "abc"],
)
def test_find_time_unit_incorrect(period: str) -> None:
    with pytest.raises(RuntimeError, match="could not find the time unit of"):
        find_time_unit(period)


###############################################
#     Tests for period_to_strftime_format     #
###############################################


def test_period_to_strftime_format_minute() -> None:
    assert period_to_strftime_format("3d12h4m") == "%Y-%m-%d %H:%M"


def test_period_to_strftime_format_month() -> None:
    assert period_to_strftime_format("3y1mo") == "%Y-%m"


def test_period_to_strftime_format_invalid() -> None:
    with pytest.raises(RuntimeError, match="could not find the time unit of invalid"):
        period_to_strftime_format("invalid")


##################################################
#     Tests for time_unit_to_strftime_format     #
##################################################


def test_time_unit_to_strftime_format_minute() -> None:
    assert time_unit_to_strftime_format("m") == "%Y-%m-%d %H:%M"


def test_time_unit_to_strftime_format_month() -> None:
    assert time_unit_to_strftime_format("mo") == "%Y-%m"


def test_time_unit_to_strftime_format_invalid() -> None:
    with pytest.raises(
        RuntimeError, match="Incorrect time unit invalid. The valid time units are:"
    ):
        time_unit_to_strftime_format("invalid")
