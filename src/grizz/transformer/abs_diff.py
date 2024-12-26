r"""Contain a ``polars.DataFrame`` transformer to compute absolute
difference between two columns."""

from __future__ import annotations

__all__ = ["AbsDiffColumnTransformer"]

import logging

import polars as pl

from grizz.transformer.columns import Base2In1OutColumnsTransformer

logger = logging.getLogger(__name__)


class AbsDiffColumnTransformer(Base2In1OutColumnsTransformer):
    r"""Implement a transformer to compute the absolute difference
    between two columns.

    Args:
        in1_col: The first input column name.
        in2_col: The seconf input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import AbsDiffColumn
    >>> transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")
    >>> transformer
    AbsDiffColumnTransformer(in1_col='col1', in2_col='col2', out_col='diff', exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": [5, 4, 3, 2, 1],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ 5    ┆ a    │
    │ 2    ┆ 4    ┆ b    │
    │ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 2    ┆ d    │
    │ 5    ┆ 1    ┆ e    │
    └──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ diff │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ str  ┆ i64  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 5    ┆ a    ┆ 4    │
    │ 2    ┆ 4    ┆ b    ┆ 2    │
    │ 3    ┆ 3    ┆ c    ┆ 0    │
    │ 4    ┆ 2    ┆ d    ┆ 2    │
    │ 5    ┆ 1    ┆ e    ┆ 4    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Computing the absolute difference between {self._in1_col} and {self._in2_col} | "
            f"out_col={self._out_col!r}"
        )
        return frame.with_columns(
            frame.select((pl.col(self._in1_col) - pl.col(self._in2_col)).abs().alias(self._out_col))
        )
