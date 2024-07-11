r"""Contain period utility functions."""

from __future__ import annotations

__all__ = ["time_unit_to_strftime_format", "find_time_unit"]

import re

STRFTIME_FORMAT = {
    "ns": "%Y-%m-%d %H:%M:%S.%f",
    "us": "%Y-%m-%d %H:%M:%S.%f",
    "ms": "%Y-%m-%d %H:%M:%S.%f",
    "s": "%Y-%m-%d %H:%M:%S",
    "m": "%Y-%m-%d %H:%M",
    "h": "%Y-%m-%d %H:%M",
    "d": "%Y-%m-%d",
    "w": "%Y %W",
    "mo": "%Y-%m",
    "q": "%Y-%m",
    "y": "%Y",
}

TIME_UNIT_TO_PERIOD_REGEX = {
    "ns": "[0-9]ns([0-9]|$)",
    "us": "[0-9]us([0-9]|$)",
    "ms": "[0-9]ms([0-9]|$)",
    "s": "[0-9]s([0-9]|$)",
    "m": "[0-9]m([0-9]|$)",
    "h": "[0-9]h([0-9]|$)",
    "d": "[0-9]d([0-9]|$)",
    "w": "[0-9]w([0-9]|$)",
    "mo": "[0-9]mo([0-9]|$)",
    "q": "[0-9]q([0-9]|$)",
    "y": "[0-9]y([0-9]|$)",
}


def find_time_unit(period: str) -> str:
    r"""Find the time unit associated to a ``polars`` period.

    Args:
        period: The ``polars`` period to analyze.

    Returns:
        The found time unit.

    Raises:
        RuntimeError: if no valid time unit can be found.

    Example usage:

    ```pycon

    >>> from grizz.utils.period import find_time_unit
    >>> find_time_unit("3d12h4m")
    m
    >>> find_time_unit("3y5mo")
    mo

    ```
    """
    for unit, regex in TIME_UNIT_TO_PERIOD_REGEX.items():
        if re.compile(regex).search(period) is not None:
            return unit

    msg = f"could not find the time unit of {period}"
    raise RuntimeError(msg)


def period_to_strftime_format(period: str) -> str:
    r"""Return the default strftime format for a given period.

    Args:
        period: The ``polars`` period to analyze.

    Returns:
        The default strftime format.

    Example usage:

    ```pycon

    >>> from grizz.utils.period import period_to_strftime_format
    >>> period_to_strftime_format("1h")
    %Y-%m-%d %H:%M
    >>> period_to_strftime_format("3y1mo")
    %Y-%m

    ```
    """
    return time_unit_to_strftime_format(time_unit=find_time_unit(period))


def time_unit_to_strftime_format(time_unit: str) -> str:
    r"""Return the default strftime format for a given time unit.

    Args:
        time_unit: The time unit.

    Returns:
        The default strftime format.

    Example usage:

    ```pycon

    >>> from grizz.utils.period import time_unit_to_strftime_format
    >>> time_unit_to_strftime_format("h")
    %Y-%m-%d %H:%M
    >>> time_unit_to_strftime_format("mo")
    %Y-%m

    ```
    """
    template = STRFTIME_FORMAT.get(time_unit.lower(), None)
    if template is None:
        msg = (
            f"Incorrect time unit {time_unit}. The valid time units are: "
            f"{list(STRFTIME_FORMAT.keys())}"
        )
        raise RuntimeError(msg)
    return template
