r"""Contain utility functions to manipulate null values in
DataFrames."""

from __future__ import annotations

__all__ = ["propagate_nulls"]

import polars as pl


def propagate_nulls(frame: pl.DataFrame, frame_with_null: pl.DataFrame) -> pl.DataFrame:
    r"""Propagate the null values from ``frame_with_null`` to ``frame``.

    Args:
        frame: The input DataFrame where to add ``None`` values based
            on ``frame_with_null``.
        frame_with_null: The DataFrame with the ``None`` values to
            propagate to ``frame``.

    Returns:
        The output DataFrame.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.null import propagate_nulls
    >>> frame_with_null = pl.DataFrame(
    ...     {
    ...         "col1": [1, None, 3, float("nan"), 5],
    ...         "col2": ["1", "2", None, "4", "5"],
    ...         "col3": [10, 20, 30, None, 50],
    ...     },
    ...     schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.Int64},
    ... )
    >>> frame = frame_with_null.fill_null(99).fill_nan(99)
    >>> frame
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ f32  ┆ str  ┆ i64  │
    ╞══════╪══════╪══════╡
    │ 1.0  ┆ 1    ┆ 10   │
    │ 99.0 ┆ 2    ┆ 20   │
    │ 3.0  ┆ null ┆ 30   │
    │ 99.0 ┆ 4    ┆ 99   │
    │ 5.0  ┆ 5    ┆ 50   │
    └──────┴──────┴──────┘
    >>> out = propagate_nulls(frame=frame, frame_with_null=frame_with_null)
    >>> out
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ f32  ┆ str  ┆ i64  │
    ╞══════╪══════╪══════╡
    │ 1.0  ┆ 1    ┆ 10   │
    │ null ┆ 2    ┆ 20   │
    │ 3.0  ┆ null ┆ 30   │
    │ 99.0 ┆ 4    ┆ null │
    │ 5.0  ┆ 5    ┆ 50   │
    └──────┴──────┴──────┘

    ```
    """
    columns = frame.columns
    return (
        frame.with_columns(frame_with_null.select(pl.all().is_null().name.suffix("__@@isnull@@_")))
        .with_columns(
            pl.when(~pl.col(col + "__@@isnull@@_")).then(pl.col(col)).otherwise(None)
            for col in columns
        )
        .select(columns)
    )
