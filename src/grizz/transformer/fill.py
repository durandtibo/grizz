r"""Contain transformers to fill values."""

from __future__ import annotations

__all__ = ["FillNanTransformer", "FillNullTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_kwargs

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class FillNanTransformer(BaseInNTransformer):
    r"""Implement a transformer to fill NaN values.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns of type string.
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
        **kwargs: The keyword arguments for ``fill_nan``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import FillNan
    >>> transformer = FillNan(columns=["col1", "col4"], value=100)
    >>> transformer
    FillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), missing_policy='raise', value=100)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, None],
    ...         "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
    ...         "col3": ["a", "b", "c", "d", None],
    ...         "col4": [1.2, float("nan"), 3.2, None, 5.2],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ str  ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1.2  ┆ a    ┆ 1.2  │
    │ 2    ┆ 2.2  ┆ b    ┆ NaN  │
    │ 3    ┆ 3.2  ┆ c    ┆ 3.2  │
    │ 4    ┆ 4.2  ┆ d    ┆ null │
    │ null ┆ NaN  ┆ null ┆ 5.2  │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4  │
    │ ---  ┆ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ f64  ┆ str  ┆ f64   │
    ╞══════╪══════╪══════╪═══════╡
    │ 1    ┆ 1.2  ┆ a    ┆ 1.2   │
    │ 2    ┆ 2.2  ┆ b    ┆ 100.0 │
    │ 3    ┆ 3.2  ┆ c    ┆ 3.2   │
    │ 4    ┆ 4.2  ┆ d    ┆ null  │
    │ null ┆ NaN  ┆ null ┆ 5.2   │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
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
        logger.info(f"Filling NaN values of {len(self.find_columns(frame)):,} columns...")
        columns = self.find_common_columns(frame)
        return frame.with_columns(
            frame.select((cs.by_name(columns) & cs.float()).fill_nan(**self._kwargs))
        )


class FillNullTransformer(BaseInNTransformer):
    r"""Implement a transformer to fill null values.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns of type string.
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
        **kwargs: The keyword arguments for ``fill_null``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import FillNull
    >>> transformer = FillNull(columns=["col1", "col4"], value=100)
    >>> transformer
    FillNullTransformer(columns=('col1', 'col4'), exclude_columns=(), missing_policy='raise', value=100)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, None],
    ...         "col2": [1.2, 2.2, 3.2, 4.2, None],
    ...         "col3": ["a", "b", "c", "d", None],
    ...         "col4": [1.2, float("nan"), 3.2, None, 5.2],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ str  ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1.2  ┆ a    ┆ 1.2  │
    │ 2    ┆ 2.2  ┆ b    ┆ NaN  │
    │ 3    ┆ 3.2  ┆ c    ┆ 3.2  │
    │ 4    ┆ 4.2  ┆ d    ┆ null │
    │ null ┆ null ┆ null ┆ 5.2  │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4  │
    │ ---  ┆ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ f64  ┆ str  ┆ f64   │
    ╞══════╪══════╪══════╪═══════╡
    │ 1    ┆ 1.2  ┆ a    ┆ 1.2   │
    │ 2    ┆ 2.2  ┆ b    ┆ NaN   │
    │ 3    ┆ 3.2  ┆ c    ┆ 3.2   │
    │ 4    ┆ 4.2  ┆ d    ┆ 100.0 │
    │ 100  ┆ null ┆ null ┆ 5.2   │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
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
        logger.info(f"Filling NaN values of {len(self.find_columns(frame)):,} columns...")
        columns = self.find_common_columns(frame)
        return frame.with_columns(frame.select(cs.by_name(columns).fill_null(**self._kwargs)))
