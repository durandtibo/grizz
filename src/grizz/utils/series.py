r"""Contain utility functions for series."""

from __future__ import annotations

__all__ = ["compute_stats_boolean"]


import polars as pl


def compute_stats_boolean(series: pl.Series) -> dict[str, float]:
    r"""Compute some basic statistics about a Boolean series.

    Args:
        series: The series to analyze.

    Returns:
        The statistics about the input Boolean series.

    Raises:
        ValueError: if ``series`` is not a Boolean series.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.series import compute_stats_boolean
    >>> series = pl.Series([True, False, None, None, False, None])
    >>> compute_stats_boolean(series)
    {'num_false': 2, 'num_null': 3, 'num_true': 1, 'total': 6}

    ```
    """
    if series.dtype != pl.Boolean:
        msg = f"Incorrect dtype. Expected a Boolean series but received {series.dtype}"
        raise ValueError(msg)
    num_null = series.is_null().sum()
    num_true = series.sum()
    total = series.shape[0]
    return {
        "num_false": total - num_true - num_null,
        "num_null": num_null,
        "num_true": num_true,
        "total": total,
    }
