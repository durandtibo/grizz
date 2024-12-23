r"""Contain ``polars.DataFrame`` transformers to copy columns."""

from __future__ import annotations

__all__ = ["CopyColumnTransformer", "CopyColumnsTransformer"]

import logging
from typing import TYPE_CHECKING

import polars as pl
from coola.utils.format import repr_mapping_line

from grizz.transformer.column import BaseColumnTransformer
from grizz.transformer.columns import BaseColumnsTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class CopyColumnTransformer(BaseColumnTransformer):
    r"""Implement a ``polars.DataFrame`` to copy a column.

    Args:
        in_col: The input column name i.e. the column to copy.
        out_col: The output column name i.e. the copied column.
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

    >>> import polars as pl
    >>> from grizz.transformer import CopyColumn
    >>> transformer = CopyColumn(in_col="col1", out_col="out")
    >>> transformer
    CopyColumnTransformer(in_col='col1', out_col='out', exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 5)
    ┌──────┬──────┬──────┬──────┬─────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ out │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ --- │
    │ i64  ┆ str  ┆ str  ┆ str  ┆ i64 │
    ╞══════╪══════╪══════╪══════╪═════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    ┆ 1   │
    │ 2    ┆ 2    ┆ 2    ┆ b    ┆ 2   │
    │ 3    ┆ 3    ┆ 3    ┆ c    ┆ 3   │
    │ 4    ┆ 4    ┆ 4    ┆ d    ┆ 4   │
    │ 5    ┆ 5    ┆ 5    ┆ e    ┆ 5   │
    └──────┴──────┴──────┴──────┴─────┘

    ```
    """

    def fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Copying column {self._in_col} to {self._out_col} ...")
        self._check_input_column(frame)
        if self._in_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.transform' "
                f"because the input column ({self._in_col}) is missing"
            )
            return frame
        self._check_output_column(frame)
        return frame.with_columns(pl.col(self._in_col).alias(self._out_col))


class CopyColumnsTransformer(BaseColumnsTransformer):
    r"""Implement a transformer to copy some columns.

    Args:
        columns: The columns to copy. ``None`` means all the
            columns.
        prefix: The column name prefix for the copied columns.
        suffix: The column name suffix for the copied columns.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message is shown.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import CopyColumns
    >>> transformer = CopyColumns(columns=["col1", "col3"], prefix="", suffix="_raw")
    >>> transformer
    CopyColumnsTransformer(columns=('col1', 'col3'), prefix='', suffix='_raw', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 6)
    ┌──────┬──────┬──────┬──────┬──────────┬──────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_raw ┆ col3_raw │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---      ┆ ---      │
    │ i64  ┆ str  ┆ str  ┆ str  ┆ i64      ┆ str      │
    ╞══════╪══════╪══════╪══════╪══════════╪══════════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    ┆ 1        ┆ 1        │
    │ 2    ┆ 2    ┆ 2    ┆ b    ┆ 2        ┆ 2        │
    │ 3    ┆ 3    ┆ 3    ┆ c    ┆ 3        ┆ 3        │
    │ 4    ┆ 4    ┆ 4    ┆ d    ┆ 4        ┆ 4        │
    │ 5    ┆ 5    ┆ 5    ┆ e    ┆ 5        ┆ 5        │
    └──────┴──────┴──────┴──────┴──────────┴──────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(columns=columns, missing_policy=missing_policy)
        self._prefix = prefix
        self._suffix = suffix

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "prefix": self._prefix,
                "suffix": self._suffix,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Copying {len(self.find_columns(frame)):,} columns | prefix={self._prefix!r} | "
            f"suffix={self._suffix!r} ..."
        )
        self._check_input_columns(frame)
        columns = self.find_common_columns(frame)
        return frame.with_columns(
            frame.select(pl.col(columns)).rename(lambda name: f"{self._prefix}{name}{self._suffix}")
        )
