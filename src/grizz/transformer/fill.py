r"""Contain transformers to fill values."""

from __future__ import annotations

__all__ = [
    "FillNanTransformer",
    "FillNullTransformer",
    "InplaceFillNanTransformer",
    "InplaceFillNullTransformer",
]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs

from grizz.transformer.columns import BaseInNOutNTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class FillNanTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to fill NaN values.

    This transformer ignores the columns that are not of type float.

    Args:
        columns: The columns of type to convert. ``None`` means
            all the columns.
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
        **kwargs: The keyword arguments for ``fill_nan``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import FillNan
    >>> transformer = FillNan(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    >>> transformer
    FillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', value=100)
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
    shape: (5, 5)
    ┌──────┬──────┬──────┬──────┬──────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col4_out │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---      │
    │ i64  ┆ f64  ┆ str  ┆ f64  ┆ f64      │
    ╞══════╪══════╪══════╪══════╪══════════╡
    │ 1    ┆ 1.2  ┆ a    ┆ 1.2  ┆ 1.2      │
    │ 2    ┆ 2.2  ┆ b    ┆ NaN  ┆ 100.0    │
    │ 3    ┆ 3.2  ┆ c    ┆ 3.2  ┆ 3.2      │
    │ 4    ┆ 4.2  ┆ d    ┆ null ┆ null     │
    │ null ┆ NaN  ┆ null ┆ 5.2  ┆ 5.2      │
    └──────┴──────┴──────┴──────┴──────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix=prefix,
            suffix=suffix,
            exclude_columns=exclude_columns,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Filling NaN values of {len(self.find_columns(frame)):,} columns...")
        columns = self.find_common_columns(frame)
        return frame.select((cs.by_name(columns) & cs.float()).fill_nan(**self._kwargs))


class InplaceFillNanTransformer(FillNanTransformer):
    r"""Implement a transformer to fill NaN values.

    This transformer ignores the columns that are not of type float.
    ``InplaceFillNanTransformer`` is a specific implementation of
    ``FillNanTransformer`` that performs the transformation in-place.

    Args:
        columns: The columns of type to convert. ``None`` means
            all the columns.
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
    >>> from grizz.transformer import InplaceFillNan
    >>> transformer = InplaceFillNan(columns=["col1", "col4"], value=100)
    >>> transformer
    InplaceFillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), missing_policy='raise', value=100)
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
        columns: Sequence[str] | None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix="",
            suffix="",
            exclude_columns=exclude_columns,
            exist_policy="ignore",
            missing_policy=missing_policy,
            **kwargs,
        )

    def get_args(self) -> dict:
        args = super().get_args()
        for key in ["prefix", "suffix", "exist_policy"]:
            args.pop(key)
        return args


class FillNullTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to fill null values.

    Args:
        columns: The columns of type to convert. ``None`` means
            all the columns.
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
        **kwargs: The keyword arguments for ``fill_null``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import FillNull
    >>> transformer = FillNull(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    >>> transformer
    FillNullTransformer(columns=('col1', 'col4'), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', value=100)
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
    shape: (5, 6)
    ┌──────┬──────┬──────┬──────┬──────────┬──────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_out ┆ col4_out │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---      ┆ ---      │
    │ i64  ┆ f64  ┆ str  ┆ f64  ┆ i64      ┆ f64      │
    ╞══════╪══════╪══════╪══════╪══════════╪══════════╡
    │ 1    ┆ 1.2  ┆ a    ┆ 1.2  ┆ 1        ┆ 1.2      │
    │ 2    ┆ 2.2  ┆ b    ┆ NaN  ┆ 2        ┆ NaN      │
    │ 3    ┆ 3.2  ┆ c    ┆ 3.2  ┆ 3        ┆ 3.2      │
    │ 4    ┆ 4.2  ┆ d    ┆ null ┆ 4        ┆ 100.0    │
    │ null ┆ null ┆ null ┆ 5.2  ┆ 100      ┆ 5.2      │
    └──────┴──────┴──────┴──────┴──────────┴──────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix=prefix,
            suffix=suffix,
            exclude_columns=exclude_columns,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Filling NaN values of {len(self.find_columns(frame)):,} columns...")
        columns = self.find_common_columns(frame)
        return frame.select(cs.by_name(columns).fill_null(**self._kwargs))


class InplaceFillNullTransformer(FillNullTransformer):
    r"""Implement a transformer to fill null values.

    This transformer ignores the columns that are not of type float.
    ``InplaceFillNullTransformer`` is a specific implementation of
    ``FillNullTransformer`` that performs the transformation in-place.

    Args:
        columns: The columns of type to convert. ``None`` means
            all the columns.
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
    >>> from grizz.transformer import InplaceFillNull
    >>> transformer = InplaceFillNull(columns=["col1", "col4"], value=100)
    >>> transformer
    InplaceFillNullTransformer(columns=('col1', 'col4'), exclude_columns=(), missing_policy='raise', value=100)
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
    │ 2    ┆ 2.2  ┆ b    ┆ NaN   │
    │ 3    ┆ 3.2  ┆ c    ┆ 3.2   │
    │ 4    ┆ 4.2  ┆ d    ┆ 100.0 │
    │ 100  ┆ NaN  ┆ null ┆ 5.2   │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix="",
            suffix="",
            exclude_columns=exclude_columns,
            exist_policy="ignore",
            missing_policy=missing_policy,
            **kwargs,
        )

    def get_args(self) -> dict:
        args = super().get_args()
        for key in ["prefix", "suffix", "exist_policy"]:
            args.pop(key)
        return args
