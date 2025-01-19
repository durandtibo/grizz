r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = [
    "CategoricalCastTransformer",
    "InplaceCategoricalCastTransformer",
]

import logging
from typing import Any

import polars as pl

from grizz.transformer.columns import BaseIn1Out1Transformer

logger = logging.getLogger(__name__)


class CategoricalCastTransformer(BaseIn1Out1Transformer):
    r"""Implement a transformer to convert a column to categorical data
    type.

    Args:
        in_col: The input column name to cast.
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
        **kwargs: Additional arguments passed to
            ``polars.Categorical``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import CategoricalCast
    >>> transformer = CategoricalCast(in_col="col1", out_col="out")
    >>> transformer
    CategoricalCastTransformer(in_col='col1', out_col='out', exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["a", "b", "c", "d", "e"],
    ...         "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     },
    ...     schema={"col1": pl.String, "col2": pl.Float64},
    ... )
    >>> frame
    shape: (5, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ str  ┆ f64  │
    ╞══════╪══════╡
    │ a    ┆ 1.0  │
    │ b    ┆ 2.0  │
    │ c    ┆ 3.0  │
    │ d    ┆ 4.0  │
    │ e    ┆ 5.0  │
    └──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 3)
    ┌──────┬──────┬─────┐
    │ col1 ┆ col2 ┆ out │
    │ ---  ┆ ---  ┆ --- │
    │ str  ┆ f64  ┆ cat │
    ╞══════╪══════╪═════╡
    │ a    ┆ 1.0  ┆ a   │
    │ b    ┆ 2.0  ┆ b   │
    │ c    ┆ 3.0  ┆ c   │
    │ d    ┆ 4.0  ┆ d   │
    │ e    ┆ 5.0  ┆ e   │
    └──────┴──────┴─────┘

    ```
    """

    def __init__(
        self,
        in_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_col=in_col,
            out_col=out_col,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return {
            "in_col": self._in_col,
            "out_col": self._out_col,
            "exist_policy": self._exist_policy,
            "missing_policy": self._missing_policy,
        } | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Casting column {self._in_col!r} to categorical column {self._out_col!r} ...")
        return frame.with_columns(
            pl.col(self._in_col).cast(pl.Categorical(**self._kwargs)).alias(self._out_col)
        )


class InplaceCategoricalCastTransformer(CategoricalCastTransformer):
    r"""Implement a transformer to convert a column to categorical data
    type.

    ``InplaceCategoricalCastTransformer`` is a specific implementation
    of ``CategoricalCastTransformer`` that performs the transformation
    in-place.

    Args:
        col: The column name to cast.
        **kwargs: Additional arguments passed to
            ``polars.Categorical``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import InplaceCategoricalCast
    >>> transformer = InplaceCategoricalCast(col="col1")
    >>> transformer
    InplaceCategoricalCastTransformer(col='col1', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["a", "b", "c", "d", "e"],
    ...         "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     },
    ...     schema={"col1": pl.String, "col2": pl.Float64},
    ... )
    >>> frame
    shape: (5, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ str  ┆ f64  │
    ╞══════╪══════╡
    │ a    ┆ 1.0  │
    │ b    ┆ 2.0  │
    │ c    ┆ 3.0  │
    │ d    ┆ 4.0  │
    │ e    ┆ 5.0  │
    └──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ cat  ┆ f64  │
    ╞══════╪══════╡
    │ a    ┆ 1.0  │
    │ b    ┆ 2.0  │
    │ c    ┆ 3.0  │
    │ d    ┆ 4.0  │
    │ e    ┆ 5.0  │
    └──────┴──────┘

    ```
    """

    def __init__(self, col: str, missing_policy: str = "raise", **kwargs: Any) -> None:
        super().__init__(
            in_col=col, out_col=col, exist_policy="ignore", missing_policy=missing_policy, **kwargs
        )

    def get_args(self) -> dict:
        return {"col": self._in_col, "missing_policy": self._missing_policy} | self._kwargs
