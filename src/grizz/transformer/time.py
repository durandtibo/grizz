# noqa: A005
r"""Contain ``polars.DataFrame`` transformers to process columns with
time values."""

from __future__ import annotations

__all__ = ["TimeToSecondTransformer", "ToTimeTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs

from grizz.transformer.columns import BaseIn1Out1Transformer, BaseInNOutNTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

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


class ToTimeTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to convert some columns to a
    ``polars.Time`` type.

    Args:
        columns: The columns of type to convert. ``None`` means
            all the columns.
        prefix: The column name prefix for the output columns.
        suffix: The column name suffix for the output columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
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
        **kwargs: The keyword arguments for ``to_time``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ToTime
    >>> transformer = ToTime(columns=["col1"], format="%H:%M:%S", prefix="", suffix="_out")
    >>> transformer
    ToTimeTransformer(columns=('col1',), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', format='%H:%M:%S')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
    ...     }
    ... )
    >>> frame
    shape: (5, 3)
    ┌──────────┬──────┬──────────┐
    │ col1     ┆ col2 ┆ col3     │
    │ ---      ┆ ---  ┆ ---      │
    │ str      ┆ str  ┆ str      │
    ╞══════════╪══════╪══════════╡
    │ 01:01:01 ┆ 1    ┆ 01:01:01 │
    │ 02:02:02 ┆ 2    ┆ 02:02:02 │
    │ 12:00:01 ┆ 3    ┆ 12:00:01 │
    │ 18:18:18 ┆ 4    ┆ 18:18:18 │
    │ 23:59:59 ┆ 5    ┆ 23:59:59 │
    └──────────┴──────┴──────────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────────┬──────┬──────────┬──────────┐
    │ col1     ┆ col2 ┆ col3     ┆ col1_out │
    │ ---      ┆ ---  ┆ ---      ┆ ---      │
    │ str      ┆ str  ┆ str      ┆ time     │
    ╞══════════╪══════╪══════════╪══════════╡
    │ 01:01:01 ┆ 1    ┆ 01:01:01 ┆ 01:01:01 │
    │ 02:02:02 ┆ 2    ┆ 02:02:02 ┆ 02:02:02 │
    │ 12:00:01 ┆ 3    ┆ 12:00:01 ┆ 12:00:01 │
    │ 18:18:18 ┆ 4    ┆ 18:18:18 ┆ 18:18:18 │
    │ 23:59:59 ┆ 5    ┆ 23:59:59 ┆ 23:59:59 │
    └──────────┴──────┴──────────┴──────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix=prefix,
            suffix=suffix,
            exclude_columns=exclude_columns,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Converting {len(self.find_columns(frame)):,} columns to time columns...")
        columns = self.find_common_columns(frame)
        frame = frame.select(cs.by_name(columns))
        return frame.select(cs.time()).with_columns(
            frame.select((~cs.time()).str.to_time(**self._kwargs))
        )
