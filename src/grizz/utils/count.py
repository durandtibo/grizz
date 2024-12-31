r"""Contain utility functions for counting."""

from __future__ import annotations

__all__ = ["compute_nunique", "compute_temporal_count", "compute_temporal_value_counts"]

from unittest.mock import Mock

import polars as pl
from coola.utils import is_numpy_available
from coola.utils.imports import check_numpy

from grizz.utils.sorting import mixed_typed_sort
from grizz.utils.temporal import to_step_names

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


def compute_nunique(frame: pl.DataFrame) -> np.ndarray:
    r"""Return the number of unique values in each column.

    Args:
        frame: The DataFrame to analyze.

    Returns:
        An array with the number of unique values in each column.
            The shape of the array is the number of columns.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.count import compute_nunique
    >>> frame = pl.DataFrame(
    ...     {
    ...         "int": [None, 1, 0, 1],
    ...         "float": [1.2, 4.2, None, 2.2],
    ...         "str": ["A", "B", None, None],
    ...     },
    ...     schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
    ... )
    >>> count = compute_nunique(frame)
    >>> count
    array([3, 4, 3])

    ```
    """
    check_numpy()
    if (ncols := frame.shape[1]) == 0:
        return np.zeros(ncols, dtype=np.int64)
    return frame.select(pl.all().n_unique()).to_numpy()[0].astype(np.int64)


def compute_temporal_count(
    frame: pl.DataFrame,
    dt_column: str,
    period: str,
) -> tuple[np.ndarray, list[str]]:
    r"""Prepare the data to create the figure and table.

    Args:
        frame: The DataFrame to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.

    Returns:
        A tuple with the counts and the temporal steps.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from grizz.utils.count import compute_temporal_count
    >>> counts, steps = compute_temporal_count(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col1": [None, float("nan"), 0.0, 1.0, 4.2, 42.0],
    ...             "col2": [None, 1, 0, None, 2, 3],
    ...             "datetime": [
    ...                 datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
    ...             ],
    ...         },
    ...         schema={
    ...             "col1": pl.Float64,
    ...             "col2": pl.Int64,
    ...             "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...         },
    ...     ),
    ...     dt_column="datetime",
    ...     period="1mo",
    ... )
    >>> counts
    array([3, 1, 1, 1])
    >>> steps
    ['2020-01', '2020-02', '2020-03', '2020-04']

    ```
    """
    check_numpy()
    if frame.is_empty():
        return np.array([], dtype=np.int64), []

    groups = (
        frame.select(pl.col(dt_column).alias("datetime"), pl.lit(1).alias("count"))
        .sort("datetime")
        .group_by_dynamic("datetime", every=period)
    )
    steps = to_step_names(groups=groups, period=period)
    counts = groups.agg(pl.col("count").sum())["count"].to_numpy().astype(np.int64)
    return counts, steps


def compute_temporal_value_counts(
    frame: pl.DataFrame,
    column: str,
    dt_column: str,
    period: str,
    drop_nulls: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    r"""Compute the value counts for temporal windows of a given column.

    Args:
        frame: The DataFrame to analyze.
        column: The column to analyze the temporal value counts.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        drop_nulls: If ``True``, the null values are ignored.

    Returns:
        A tuple with 3 items. The first item is a 2-d array that
            indicates the number of occurrences for each value and
            time step. The first dimension represents the value and
            the second dimension represents the steps. The second item
            is the list of time steps. The third item is the list of
            string representation of the values.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from grizz.utils.count import compute_temporal_value_counts
    >>> counts, steps, values = compute_temporal_value_counts(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col1": [None, 1.0, 0.0, 1.0, 4.2, 42.0],
    ...             "col2": [None, 1, 0, None, 2, 3],
    ...             "datetime": [
    ...                 datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
    ...             ],
    ...         },
    ...         schema={
    ...             "col1": pl.Float64,
    ...             "col2": pl.Int64,
    ...             "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...         },
    ...     ),
    ...     column="col1",
    ...     dt_column="datetime",
    ...     period="1mo",
    ... )
    >>> counts
    array([[1, 0, 0, 0],
           [1, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [1, 0, 0, 0]])
    >>> steps
    ['2020-01', '2020-02', '2020-03', '2020-04']
    >>> values
    ['0.0', '1.0', '4.2', '42.0', 'null']

    ```
    """
    check_numpy()
    if frame.is_empty():
        return np.zeros((0, 0), dtype=np.int64), [], []

    frame = frame.select(pl.col(dt_column).alias("__datetime__"), pl.col(column).alias("value"))
    if drop_nulls:
        frame = frame.drop_nulls()

    groups = frame.sort(["__datetime__", "value"]).group_by_dynamic("__datetime__", every=period)
    steps = to_step_names(groups=groups, period="1mo")
    frame_counts = (
        groups.agg(pl.col("value").value_counts())
        .explode("value")
        .unnest("value")
        .pivot(on="value", index="__datetime__", values="count")
        .drop("__datetime__")
    )
    frame_counts = frame_counts.select(mixed_typed_sort(frame_counts.columns))
    counts = frame_counts.fill_null(0.0).to_numpy().astype(np.int64).transpose()
    return counts, steps, list(frame_counts.columns)
