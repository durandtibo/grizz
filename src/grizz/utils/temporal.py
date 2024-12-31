r"""Contain utility functions to do temporal transformations."""

from __future__ import annotations

__all__ = ["compute_temporal_stats", "to_step_names", "to_temporal_frames"]


import polars as pl

from grizz.utils.interval import interval_to_strftime_format


def compute_temporal_stats(
    frame: pl.DataFrame,
    column: str,
    dt_column: str,
    period: str,
) -> pl.DataFrame:
    r"""Return a DataFrame with stats for each temporal window.

    Args:
        frame: The DataFrame to analyze.
        column: The column to analyze.
        dt_column: The datetime column used to create the temporal
            DataFrames.
        period: The temporal period e.g. monthly or daily.

    Returns:
        A DataFrame with stats for each temporal window.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from grizz.utils.temporal import compute_temporal_stats
    >>> stats = compute_temporal_stats(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col": [1.2, 4.2, 0.0, 1.0, 4.2, 42.0],
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
    ...             "col": pl.Float64,
    ...             "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...         },
    ...     ),
    ...     column="col",
    ...     dt_column="datetime",
    ...     period="1mo",
    ... )
    >>> stats
    shape: (4, 16)
    ┌─────────────────────────┬───────┬─────────┬──────┬───┬──────┬──────┬──────┬──────┐
    │ step                    ┆ count ┆ nunique ┆ mean ┆ … ┆ q90  ┆ q95  ┆ q99  ┆ max  │
    │ ---                     ┆ ---   ┆ ---     ┆ ---  ┆   ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ datetime[μs, UTC]       ┆ i64   ┆ i64     ┆ f64  ┆   ┆ f64  ┆ f64  ┆ f64  ┆ f64  │
    ╞═════════════════════════╪═══════╪═════════╪══════╪═══╪══════╪══════╪══════╪══════╡
    │ 2020-01-01 00:00:00 UTC ┆ 3     ┆ 3       ┆ 1.8  ┆ … ┆ 4.2  ┆ 4.2  ┆ 4.2  ┆ 4.2  │
    │ 2020-02-01 00:00:00 UTC ┆ 1     ┆ 1       ┆ 1.0  ┆ … ┆ 1.0  ┆ 1.0  ┆ 1.0  ┆ 1.0  │
    │ 2020-03-01 00:00:00 UTC ┆ 1     ┆ 1       ┆ 4.2  ┆ … ┆ 4.2  ┆ 4.2  ┆ 4.2  ┆ 4.2  │
    │ 2020-04-01 00:00:00 UTC ┆ 1     ┆ 1       ┆ 42.0 ┆ … ┆ 42.0 ┆ 42.0 ┆ 42.0 ┆ 42.0 │
    └─────────────────────────┴───────┴─────────┴──────┴───┴──────┴──────┴──────┴──────┘

    ```
    """
    return (
        frame.select(column, dt_column)
        .sort(dt_column)
        .group_by_dynamic(dt_column, every=period)
        .agg(
            pl.col(column).len().cast(pl.Int64).alias("count"),
            pl.col(column).n_unique().cast(pl.Int64).alias("nunique"),
            pl.col(column).mean().cast(pl.Float64).alias("mean"),
            pl.col(column).std().cast(pl.Float64).alias("std"),
            pl.col(column).min().cast(pl.Float64).alias("min"),
            pl.col(column).quantile(0.01).cast(pl.Float64).alias("q01"),
            pl.col(column).quantile(0.05).cast(pl.Float64).alias("q05"),
            pl.col(column).quantile(0.1).cast(pl.Float64).alias("q10"),
            pl.col(column).quantile(0.25).cast(pl.Float64).alias("q25"),
            pl.col(column).median().cast(pl.Float64).alias("median"),
            pl.col(column).quantile(0.75).cast(pl.Float64).alias("q75"),
            pl.col(column).quantile(0.9).cast(pl.Float64).alias("q90"),
            pl.col(column).quantile(0.95).cast(pl.Float64).alias("q95"),
            pl.col(column).quantile(0.99).cast(pl.Float64).alias("q99"),
            pl.col(column).max().cast(pl.Float64).alias("max"),
        )
        .rename({dt_column: "step"})
    )


def to_temporal_frames(
    frame: pl.DataFrame,
    dt_column: str,
    period: str,
) -> tuple[list[pl.DataFrame], list[str]]:
    r"""Return a list of temporal DataFrames and the associated time
    steps.

    Args:
        frame: The DataFrame to analyze.
        dt_column: The datetime column used to create the temporal
            DataFrames.
        period: The temporal period e.g. monthly or daily.

    Returns:
        A tuple with the counts and the temporal steps.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from grizz.utils.temporal import to_temporal_frames
    >>> frames, steps = to_temporal_frames(
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
    >>> frames
    [shape: (3, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ null ┆ null ┆ 2020-01-03 00:00:00 UTC │
    │ NaN  ┆ 1    ┆ 2020-01-04 00:00:00 UTC │
    │ 0.0  ┆ 0    ┆ 2020-01-05 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘, shape: (1, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ 1.0  ┆ null ┆ 2020-02-03 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘, shape: (1, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ 4.2  ┆ 2    ┆ 2020-03-03 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘, shape: (1, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ 42.0 ┆ 3    ┆ 2020-04-03 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘]
    >>> steps
    ['2020-01', '2020-02', '2020-03', '2020-04']

    ```
    """
    if frame.is_empty():
        return [], []

    groups = frame.sort(dt_column).group_by_dynamic(dt_column, every=period)
    steps = to_step_names(groups=groups, period=period)
    frames = [frame for _, frame in groups]
    return frames, steps


def to_step_names(groups: pl.GroupBy, period: str) -> list[str]:
    r"""Return the name of each step.

    Args:
        groups: The grouped DataFrame by step.
        period: The temporal period e.g. monthly or daily.

    Returns:
        A list that contains the name of each step.

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from grizz.utils.temporal import to_step_names
    >>> groups = (
    ...     pl.DataFrame(
    ...         {
    ...             "col": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
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
    ...             "col": pl.Float64,
    ...             "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...         },
    ...     )
    ...     .sort("datetime")
    ...     .group_by_dynamic("datetime", every="1mo")
    ... )
    >>> steps = to_step_names(groups=groups, period="1mo")
    >>> steps
    ['2020-01', '2020-02', '2020-03', '2020-04']

    ```
    """
    format_dt = interval_to_strftime_format(period)
    return [name[0].strftime(format_dt) for name, _ in groups]
