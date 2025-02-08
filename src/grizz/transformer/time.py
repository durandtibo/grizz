# noqa: A005
r"""Contain ``polars.DataFrame`` transformers to process columns with
time values."""

from __future__ import annotations

__all__ = ["TimeToSecondTransformer"]

import logging

import polars as pl

from grizz.transformer.columns import BaseIn1Out1Transformer
from grizz.transformer.utils import get_classname, message_skip_fit

logger = logging.getLogger(__name__)


class TimeToSecondTransformer(BaseIn1Out1Transformer):
    r"""Implement a transformer to convert a column with time values to
    seconds.

    Args:
        in_col: The input column with the time value to convert.
        out_col: The output column with the time in seconds.
        exist_policy: The policy on how to handle existing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column already exist.
            If ``'warn'``, a warning is raised if at least one column
            already exist and the existing columns are overwritten.
            If ``'ignore'``, the existing columns are overwritten and
            no warning message appears.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message appears.

    Example usage:

    ```pycon

    >>> import datetime
    >>> import polars as pl
    >>> from grizz.transformer import TimeToSecond
    >>> transformer = TimeToSecond(in_col="time", out_col="second")
    >>> transformer
    TimeToSecondTransformer(in_col='time', out_col='second', exist_policy='raise', missing_policy='raise')
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

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Converting time column {self._in_col!r} to seconds {self._out_col!r} ...")
        return frame.with_columns(
            frame.select(
                pl.col(self._in_col)
                .cast(pl.Duration)
                .dt.total_microseconds()
                .truediv(1e6)
                .alias(self._out_col)
            )
        )
