r"""Contain period utility functions."""

from __future__ import annotations

import re

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
