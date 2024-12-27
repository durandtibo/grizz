r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = [
    "CastTransformer",
    "DecimalCastTransformer",
    "FloatCastTransformer",
    "IntegerCastTransformer",
]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_kwargs, str_size_diff

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class CastTransformer(BaseInNTransformer):
    r"""Implement a transformer to convert some columns to a new data
    type.

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
    >>> from grizz.transformer import Cast
    >>> transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    >>> transformer
    CastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), missing_policy='raise')
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
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i32  ┆ str  ┆ i32  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        dtype: type[pl.DataType],
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns, exclude_columns=exclude_columns, missing_policy=missing_policy
        )
        self._dtype = dtype
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "dtype": self._dtype,
                "exclude_columns": self._exclude_columns,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args}{str_kwargs(self._kwargs)})"

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Casting {len(self.find_columns(frame)):,} columns to {self._dtype}...")
        columns = self.find_common_columns(frame)
        out = self._transform_frame(frame, columns)
        logger.info(str_size_diff(orig=frame.estimated_size(), final=out.estimated_size()))
        return out

    def _transform_frame(self, frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        return frame.with_columns(
            frame.select(cs.by_name(columns).cast(self._dtype, **self._kwargs))
        )


class DecimalCastTransformer(CastTransformer):
    r"""Implement a transformer to convert columns of type decimal to a
    new data type.

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
    >>> from grizz.transformer import DecimalCast
    >>> transformer = DecimalCast(columns=["col1", "col2"], dtype=pl.Float32)
    >>> transformer
    DecimalCastTransformer(columns=('col1', 'col2'), dtype=Float32, exclude_columns=(), missing_policy='raise')
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

    def _transform_frame(self, frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        return frame.with_columns(
            frame.select((cs.by_name(columns) & cs.decimal()).cast(self._dtype, **self._kwargs))
        )


class FloatCastTransformer(CastTransformer):
    r"""Implement a transformer to convert columns of type float to a new
    data type.

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
    >>> from grizz.transformer import FloatCast
    >>> transformer = FloatCast(columns=["col1", "col2"], dtype=pl.Int32)
    >>> transformer
    FloatCastTransformer(columns=('col1', 'col2'), dtype=Int32, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     },
    ...     schema={
    ...         "col1": pl.Int64,
    ...         "col2": pl.Float64,
    ...         "col3": pl.Float64,
    ...         "col4": pl.String,
    ...     },
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ f64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1.0  ┆ 1.0  ┆ a    │
    │ 2    ┆ 2.0  ┆ 2.0  ┆ b    │
    │ 3    ┆ 3.0  ┆ 3.0  ┆ c    │
    │ 4    ┆ 4.0  ┆ 4.0  ┆ d    │
    │ 5    ┆ 5.0  ┆ 5.0  ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i32  ┆ f64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1.0  ┆ a    │
    │ 2    ┆ 2    ┆ 2.0  ┆ b    │
    │ 3    ┆ 3    ┆ 3.0  ┆ c    │
    │ 4    ┆ 4    ┆ 4.0  ┆ d    │
    │ 5    ┆ 5    ┆ 5.0  ┆ e    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def _transform_frame(self, frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        return frame.with_columns(
            frame.select((cs.by_name(columns) & cs.float()).cast(self._dtype, **self._kwargs))
        )


class IntegerCastTransformer(CastTransformer):
    r"""Implement a transformer to convert columns of type integer to a
    new data type.

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
    >>> from grizz.transformer import IntegerCast
    >>> transformer = IntegerCast(columns=["col1", "col2"], dtype=pl.Float32)
    >>> transformer
    IntegerCastTransformer(columns=('col1', 'col2'), dtype=Float32, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "col3": [1, 2, 3, 4, 5],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     },
    ...     schema={
    ...         "col1": pl.Int64,
    ...         "col2": pl.Float64,
    ...         "col3": pl.Int64,
    ...         "col4": pl.String,
    ...     },
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1.0  ┆ 1    ┆ a    │
    │ 2    ┆ 2.0  ┆ 2    ┆ b    │
    │ 3    ┆ 3.0  ┆ 3    ┆ c    │
    │ 4    ┆ 4.0  ┆ 4    ┆ d    │
    │ 5    ┆ 5.0  ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ f32  ┆ f64  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1.0  ┆ 1.0  ┆ 1    ┆ a    │
    │ 2.0  ┆ 2.0  ┆ 2    ┆ b    │
    │ 3.0  ┆ 3.0  ┆ 3    ┆ c    │
    │ 4.0  ┆ 4.0  ┆ 4    ┆ d    │
    │ 5.0  ┆ 5.0  ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def _transform_frame(self, frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        return frame.with_columns(
            frame.select((cs.by_name(columns) & cs.integer()).cast(self._dtype, **self._kwargs))
        )
