r"""Contain transformers to concatenate columns."""

from __future__ import annotations

__all__ = ["ConcatColumnsTransformer"]

import logging

import polars as pl
import polars.selectors as cs

from grizz.transformer.columns import BaseInNOut1Transformer

logger = logging.getLogger(__name__)


class ConcatColumnsTransformer(BaseInNOut1Transformer):
    r"""Implement a transformer to concatenate columns into a new column.

    Args:
        columns: The columns to concatenate. The columns should have
            the same type or compatible types. If ``None``,
            it processes all the columns.
        out_col: The output column.
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

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ConcatColumns
    >>> transformer = ConcatColumns(columns=["col1", "col2", "col3"], out_col="col")
    >>> transformer
    ConcatColumnsTransformer(columns=('col1', 'col2', 'col3'), out_col='col', exclude_columns=(), exist_policy='raise', missing_policy='raise')
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

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Concatenating {len(self.find_columns(frame)):,} columns to {self._out_col!r} ..."
        )
        columns = self.find_common_columns(frame)
        return frame.with_columns(
            frame.select(pl.concat_list(cs.by_name(columns).alias(self._out_col)))
        )
