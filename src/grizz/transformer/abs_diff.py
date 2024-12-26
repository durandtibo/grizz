r"""Contain a ``polars.DataFrame`` transformer to compute absolute
difference between two columns."""

from __future__ import annotations

__all__ = ["AbsDiffColumnTransformer"]

import logging

import polars as pl
from coola.utils.format import repr_mapping_line

from grizz.transformer.base import BaseTransformer
from grizz.utils.column import (
    check_column_exist_policy,
    check_column_missing_policy,
    check_existing_column,
    check_missing_column,
)

logger = logging.getLogger(__name__)


class AbsDiffColumnTransformer(BaseTransformer):
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

    def __init__(
        self,
        in1_col: str,
        in2_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        self._in1_col = in1_col
        self._in2_col = in2_col
        self._out_col = out_col

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy
        check_column_missing_policy(missing_policy)
        self._missing_policy = missing_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "in1_col": self._in1_col,
                "in2_col": self._in2_col,
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

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Computing the absolute difference between {self._in1_col} and {self._in2_col} | "
            f"out_col={self._out_col!r}"
        )
        self._check_input_columns(frame)
        if self._in1_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.transform' "
                f"because the input column ({self._in1_col}) is missing"
            )
            return frame
        if self._in2_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.transform' "
                f"because the input column ({self._in2_col}) is missing"
            )
            return frame
        self._check_output_column(frame)
        return frame.with_columns(
            frame.select((pl.col(self._in1_col) - pl.col(self._in2_col)).abs().alias(self._out_col))
        )

    def _check_input_columns(self, frame: pl.DataFrame) -> None:
        r"""Check if any of the input columns is missing.

        Args:
            frame: The input DataFrame to check.
        """
        check_missing_column(frame, column=self._in1_col, missing_policy=self._missing_policy)
        check_missing_column(frame, column=self._in2_col, missing_policy=self._missing_policy)

    def _check_output_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input DataFrame to check.
        """
        check_existing_column(frame, column=self._out_col, exist_policy=self._exist_policy)
