r"""Contain ``polars.DataFrame`` transformers to convert decimal columns
to a new data type."""

from __future__ import annotations

__all__ = ["DecimalCastTransformer", "InplaceDecimalCastTransformer"]

import logging
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from grizz.transformer.cast.universal import CastTransformer, InplaceCastTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class DecimalCastTransformer(CastTransformer):
    r"""Implement a transformer to convert decimal columns to a new data
    type.

    Args:
        columns: The columns to convert. ``None`` means all the
            columns.
        dtype: The target data type.
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
        **kwargs: The keyword arguments for ``cast``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import DecimalCast
    >>> transformer = DecimalCast(
    ...     columns=["col1", "col2"], dtype=pl.Float32, prefix="", suffix="_out"
    ... )
    >>> transformer
    DecimalCastTransformer(columns=('col1', 'col2'), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', dtype=Float32)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     },
    ...     schema={
    ...         "col1": pl.Int64,
    ...         "col2": pl.Decimal,
    ...         "col3": pl.Decimal,
    ...         "col4": pl.String,
    ...     },
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────────────┬──────────────┬──────┐
    │ col1 ┆ col2         ┆ col3         ┆ col4 │
    │ ---  ┆ ---          ┆ ---          ┆ ---  │
    │ i64  ┆ decimal[*,0] ┆ decimal[*,0] ┆ str  │
    ╞══════╪══════════════╪══════════════╪══════╡
    │ 1    ┆ 1            ┆ 1            ┆ a    │
    │ 2    ┆ 2            ┆ 2            ┆ b    │
    │ 3    ┆ 3            ┆ 3            ┆ c    │
    │ 4    ┆ 4            ┆ 4            ┆ d    │
    │ 5    ┆ 5            ┆ 5            ┆ e    │
    └──────┴──────────────┴──────────────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 5)
    ┌──────┬──────────────┬──────────────┬──────┬──────────┐
    │ col1 ┆ col2         ┆ col3         ┆ col4 ┆ col2_out │
    │ ---  ┆ ---          ┆ ---          ┆ ---  ┆ ---      │
    │ i64  ┆ decimal[*,0] ┆ decimal[*,0] ┆ str  ┆ f32      │
    ╞══════╪══════════════╪══════════════╪══════╪══════════╡
    │ 1    ┆ 1            ┆ 1            ┆ a    ┆ 1.0      │
    │ 2    ┆ 2            ┆ 2            ┆ b    ┆ 2.0      │
    │ 3    ┆ 3            ┆ 3            ┆ c    ┆ 3.0      │
    │ 4    ┆ 4            ┆ 4            ┆ d    ┆ 4.0      │
    │ 5    ┆ 5            ┆ 5            ┆ e    ┆ 5.0      │
    └──────┴──────────────┴──────────────┴──────┴──────────┘

    ```
    """

    def _cast(self, frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        return frame.select((cs.by_name(columns) & cs.decimal()).cast(self._dtype, **self._kwargs))


class InplaceDecimalCastTransformer(InplaceCastTransformer):
    r"""Implement a transformer to convert decimal columns to a new data
    type.

    ``InplaceCastTransformer`` is a specific implementation of
    ``CastTransformer`` that performs the transformation in-place.

    Args:
        columns: The columns to convert. ``None`` means all the
            columns.
        dtype: The target data type.
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
        **kwargs: The keyword arguments for ``cast``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import InplaceDecimalCast
    >>> transformer = InplaceDecimalCast(columns=["col1", "col2"], dtype=pl.Float32)
    >>> transformer
    InplaceDecimalCastTransformer(columns=('col1', 'col2'), exclude_columns=(), missing_policy='raise', dtype=Float32)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     },
    ...     schema={
    ...         "col1": pl.Int64,
    ...         "col2": pl.Decimal,
    ...         "col3": pl.Decimal,
    ...         "col4": pl.String,
    ...     },
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────────────┬──────────────┬──────┐
    │ col1 ┆ col2         ┆ col3         ┆ col4 │
    │ ---  ┆ ---          ┆ ---          ┆ ---  │
    │ i64  ┆ decimal[*,0] ┆ decimal[*,0] ┆ str  │
    ╞══════╪══════════════╪══════════════╪══════╡
    │ 1    ┆ 1            ┆ 1            ┆ a    │
    │ 2    ┆ 2            ┆ 2            ┆ b    │
    │ 3    ┆ 3            ┆ 3            ┆ c    │
    │ 4    ┆ 4            ┆ 4            ┆ d    │
    │ 5    ┆ 5            ┆ 5            ┆ e    │
    └──────┴──────────────┴──────────────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────────────┬──────┐
    │ col1 ┆ col2 ┆ col3         ┆ col4 │
    │ ---  ┆ ---  ┆ ---          ┆ ---  │
    │ i64  ┆ f32  ┆ decimal[*,0] ┆ str  │
    ╞══════╪══════╪══════════════╪══════╡
    │ 1    ┆ 1.0  ┆ 1            ┆ a    │
    │ 2    ┆ 2.0  ┆ 2            ┆ b    │
    │ 3    ┆ 3.0  ┆ 3            ┆ c    │
    │ 4    ┆ 4.0  ┆ 4            ┆ d    │
    │ 5    ┆ 5.0  ┆ 5            ┆ e    │
    └──────┴──────┴──────────────┴──────┘

    ```
    """

    def _cast(self, frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        return frame.select((cs.by_name(columns) & cs.decimal()).cast(self._dtype, **self._kwargs))
