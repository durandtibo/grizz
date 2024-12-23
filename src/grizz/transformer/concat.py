r"""Contain transformers to concatenate columns."""

from __future__ import annotations

__all__ = ["ConcatColumnsTransformer"]

import logging
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseColumnsTransformer
from grizz.utils.column import check_column_exist_policy, check_existing_columns

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class ConcatColumnsTransformer(BaseColumnsTransformer):
    r"""Implement a transformer to concatenate columns into a new column.

    Args:
        columns: The columns to concatenate. The column should have
            the same type or compatible types.
        out_col: The output column.
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
            no warning message is shown.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ConcatColumns
    >>> transformer = ConcatColumns(columns=["col1", "col2", "col3"], out_col="col")
    >>> transformer
    ConcatColumnsTransformer(columns=('col1', 'col2', 'col3'), out_col='col', exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [11, 12, 13, 14, 15],
    ...         "col2": [21, 22, 23, 24, 25],
    ...         "col3": [31, 32, 33, 34, 35],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 11   ┆ 21   ┆ 31   ┆ a    │
    │ 12   ┆ 22   ┆ 32   ┆ b    │
    │ 13   ┆ 23   ┆ 33   ┆ c    │
    │ 14   ┆ 24   ┆ 34   ┆ d    │
    │ 15   ┆ 25   ┆ 35   ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 5)
    ┌──────┬──────┬──────┬──────┬──────────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col          │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---          │
    │ i64  ┆ i64  ┆ i64  ┆ str  ┆ list[i64]    │
    ╞══════╪══════╪══════╪══════╪══════════════╡
    │ 11   ┆ 21   ┆ 31   ┆ a    ┆ [11, 21, 31] │
    │ 12   ┆ 22   ┆ 32   ┆ b    ┆ [12, 22, 32] │
    │ 13   ┆ 23   ┆ 33   ┆ c    ┆ [13, 23, 33] │
    │ 14   ┆ 24   ┆ 34   ┆ d    ┆ [14, 24, 34] │
    │ 15   ┆ 25   ┆ 35   ┆ e    ┆ [15, 25, 35] │
    └──────┴──────┴──────┴──────┴──────────────┘


    ```
    """

    def __init__(
        self,
        columns: Sequence[str],
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(columns=columns, missing_policy=missing_policy)
        self._out_col = out_col

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "out_col": self._out_col,
                "exist_policy": self._exist_policy,
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
            f"Concatenating {len(self.find_columns(frame)):,} columns to {self._out_col}..."
        )
        self._check_input_columns(frame)
        self._check_output_column(frame)
        columns = self.find_common_columns(frame)
        return frame.with_columns(
            frame.select(pl.concat_list(cs.by_name(columns).alias(self._out_col)))
        )

    def _check_output_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input DataFrame to check.
        """
        check_existing_columns(frame, columns=[self._out_col], exist_policy=self._exist_policy)
