r"""Contain transformers to max rows or columns."""

from __future__ import annotations

__all__ = ["MaxHorizontalTransformer"]

import logging

import polars as pl

from grizz.transformer.columns import BaseInNOut1Transformer

logger = logging.getLogger(__name__)


class MaxHorizontalTransformer(BaseInNOut1Transformer):
    r"""Implement a transformer to get the maximum value horizontally
    across columns and store the result in a column.

    Args:
        columns: The columns the maximum value horizontally.
            The columns should be compatible.
            If ``None``, it processes all the columns.
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
    >>> from grizz.transformer import MaxHorizontal
    >>> transformer = MaxHorizontal(columns=["col1", "col2", "col3"], out_col="col")
    >>> transformer
    MaxHorizontalTransformer(columns=('col1', 'col2', 'col3'), out_col='col', exclude_columns=(), exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [9, 5, 4, 9, 6],
    ...         "col2": [8, 0, 1, 8, 9],
    ...         "col3": [0, 4, 8, 7, 0],
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
    │ 9    ┆ 8    ┆ 0    ┆ a    │
    │ 5    ┆ 0    ┆ 4    ┆ b    │
    │ 4    ┆ 1    ┆ 8    ┆ c    │
    │ 9    ┆ 8    ┆ 7    ┆ d    │
    │ 6    ┆ 9    ┆ 0    ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 5)
    ┌──────┬──────┬──────┬──────┬─────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ --- │
    │ i64  ┆ i64  ┆ i64  ┆ str  ┆ i64 │
    ╞══════╪══════╪══════╪══════╪═════╡
    │ 9    ┆ 8    ┆ 0    ┆ a    ┆ 9   │
    │ 5    ┆ 0    ┆ 4    ┆ b    ┆ 5   │
    │ 4    ┆ 1    ┆ 8    ┆ c    ┆ 8   │
    │ 9    ┆ 8    ┆ 7    ┆ d    ┆ 9   │
    │ 6    ┆ 9    ┆ 0    ┆ e    ┆ 9   │
    └──────┴──────┴──────┴──────┴─────┘

    ```
    """

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        cols = self.find_columns(frame)
        logger.info(
            f"Getting the maximum value across {len(cols):,} columns: {self.find_columns(frame)} "
            f"| out_col={self._out_col!r}"
        )
        columns = self.find_common_columns(frame)
        return frame.with_columns(pl.max_horizontal(columns).alias(self._out_col))
