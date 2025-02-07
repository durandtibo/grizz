r"""Contain ``polars.DataFrame`` transformers to process columns with
time values."""

from __future__ import annotations

__all__ = ["InplaceStringToTimeTransformer", "StringToTimeTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs

from grizz.transformer.columns import BaseInNOutNTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class StringToTimeTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to convert some string columns to
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
        **kwargs: The keyword arguments for
            ``polars.Expr.str.to_time``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import StringToTime
    >>> transformer = StringToTime(
    ...     columns=["col1"], format="%H:%M:%S", prefix="", suffix="_out"
    ... )
    >>> transformer
    StringToTimeTransformer(columns=('col1',), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', format='%H:%M:%S')
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
        columns = self.find_common_columns(frame)
        logger.info(f"Converting {len(columns):,} string columns to time columns...")
        return frame.select((cs.by_name(columns) & cs.string()).str.to_time(**self._kwargs))


class InplaceStringToTimeTransformer(StringToTimeTransformer):
    r"""Implement a transformer to convert some string columns to a
    ``polars.Time`` type.

    Args:
        columns: The columns of type to convert. ``None`` means
            all the columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message appears.
        **kwargs: The keyword arguments for
            ``polars.Expr.str.to_time``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import InplaceStringToTime
    >>> transformer = InplaceStringToTime(columns=["col1"], format="%H:%M:%S")
    >>> transformer
    InplaceStringToTimeTransformer(columns=('col1',), exclude_columns=(), missing_policy='raise', format='%H:%M:%S')
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
    shape: (5, 3)
    ┌──────────┬──────┬──────────┐
    │ col1     ┆ col2 ┆ col3     │
    │ ---      ┆ ---  ┆ ---      │
    │ time     ┆ str  ┆ str      │
    ╞══════════╪══════╪══════════╡
    │ 01:01:01 ┆ 1    ┆ 01:01:01 │
    │ 02:02:02 ┆ 2    ┆ 02:02:02 │
    │ 12:00:01 ┆ 3    ┆ 12:00:01 │
    │ 18:18:18 ┆ 4    ┆ 18:18:18 │
    │ 23:59:59 ┆ 5    ┆ 23:59:59 │
    └──────────┴──────┴──────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix="",
            suffix="",
            exclude_columns=exclude_columns,
            exist_policy="ignore",
            missing_policy=missing_policy,
            **kwargs,
        )

    def get_args(self) -> dict:
        args = super().get_args()
        for key in ["prefix", "suffix", "exist_policy"]:
            args.pop(key)
        return args
