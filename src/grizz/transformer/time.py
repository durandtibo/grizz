r"""Contain ``polars.DataFrame`` transformers to process columns with
time values."""

from __future__ import annotations

__all__ = ["TimeToSecondTransformer"]


import polars as pl

from grizz.transformer.base import BaseTransformer


class TimeToSecondTransformer(BaseTransformer):
    r"""Implement a transformer to convert a column with time values to
    seconds.

    Args:
        in_col: The input column with the time value to convert.
        out_col: The output column with the time in seconds.

    Example usage:

    ```pycon

    >>> import datetime
    >>> import polars as pl
    >>> from grizz.transformer import TimeToSecond
    >>> transformer = TimeToSecond(in_col="time", out_col="second")
    >>> transformer
    TimeToSecondTransformer(in_col=time, out_col=second)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "time": [
    ...             datetime.time(0, 0, 1, 890000),
    ...             datetime.time(0, 1, 1, 890000),
    ...             datetime.time(1, 1, 1, 890000),
    ...             datetime.time(0, 19, 19, 890000),
    ...             datetime.time(19, 19, 19, 890000),
    ...         ],
    ...         "col": ["a", "b", "c", "d", "e"],
    ...     },
    ...     schema={"time": pl.Time, "col": pl.String},
    ... )
    >>> frame
    shape: (5, 2)
    ┌──────────────┬─────┐
    │ time         ┆ col │
    │ ---          ┆ --- │
    │ time         ┆ str │
    ╞══════════════╪═════╡
    │ 00:00:01.890 ┆ a   │
    │ 00:01:01.890 ┆ b   │
    │ 01:01:01.890 ┆ c   │
    │ 00:19:19.890 ┆ d   │
    │ 19:19:19.890 ┆ e   │
    └──────────────┴─────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 3)
    ┌──────────────┬─────┬──────────┐
    │ time         ┆ col ┆ second   │
    │ ---          ┆ --- ┆ ---      │
    │ time         ┆ str ┆ f64      │
    ╞══════════════╪═════╪══════════╡
    │ 00:00:01.890 ┆ a   ┆ 1.89     │
    │ 00:01:01.890 ┆ b   ┆ 61.89    │
    │ 01:01:01.890 ┆ c   ┆ 3661.89  │
    │ 00:19:19.890 ┆ d   ┆ 1159.89  │
    │ 19:19:19.890 ┆ e   ┆ 69559.89 │
    └──────────────┴─────┴──────────┘

    ```
    """

    def __init__(self, in_col: str, out_col: str) -> None:
        self._in_col = in_col
        self._out_col = out_col

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(in_col={self._in_col}, out_col={self._out_col})"

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            frame.select(
                pl.col(self._in_col)
                .cast(pl.Duration)
                .dt.total_microseconds()
                .truediv(1e6)
                .alias(self._out_col)
            )
        )
