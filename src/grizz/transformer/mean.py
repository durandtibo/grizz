r"""Contain transformers to mean rows or columns."""

from __future__ import annotations

__all__ = ["MeanHorizontalTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from grizz.transformer.columns import BaseInNOut1Transformer
from grizz.transformer.utils import get_classname, message_skip_fit

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class MeanHorizontalTransformer(BaseInNOut1Transformer):
    r"""Implement a transformer to get the mean value horizontally across
    columns and store the result in a column.

    Args:
        columns: The columns the mean value horizontally.
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
        **kwargs: Additional arguments passed to
            ``polars.mean_horizontal``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import MeanHorizontal
    >>> transformer = MeanHorizontal(columns=["col1", "col2", "col3"], out_col="col")
    >>> transformer
    MeanHorizontalTransformer(columns=('col1', 'col2', 'col3'), out_col='col', exclude_columns=(), exist_policy='raise', missing_policy='raise')
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
    ┌──────┬──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col  │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ str  ┆ f64  │
    ╞══════╪══════╪══════╪══════╪══════╡
    │ 11   ┆ 21   ┆ 31   ┆ a    ┆ 21.0 │
    │ 12   ┆ 22   ┆ 32   ┆ b    ┆ 22.0 │
    │ 13   ┆ 23   ┆ 33   ┆ c    ┆ 23.0 │
    │ 14   ┆ 24   ┆ 34   ┆ d    ┆ 24.0 │
    │ 15   ┆ 25   ┆ 35   ┆ e    ┆ 25.0 │
    └──────┴──────┴──────┴──────┴──────┘


    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        out_col: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            out_col=out_col,
            exclude_columns=exclude_columns,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        columns = self.find_common_columns(frame)
        logger.info(
            f"Getting the mean value across {len(columns):,} columns: {columns} "
            f"| out_col={self._out_col!r}"
        )
        if not columns:
            return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(self._out_col))
        return frame.with_columns(pl.mean_horizontal(columns, **self._kwargs).alias(self._out_col))
