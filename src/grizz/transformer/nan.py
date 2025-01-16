r"""Contain transformers to drop columns or rows with NaN values."""

from __future__ import annotations

__all__ = ["DropNanColumnTransformer", "DropNanRowTransformer"]

import logging
from itertools import compress
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_kwargs, str_shape_diff

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class DropNanColumnTransformer(BaseInNTransformer):
    r"""Implement a transformer to remove the columns that have too many
    NaN values.

    Args:
        columns: The columns to convert. ``None`` means all the
            columns.
        threshold: The maximum percentage of NaN values to keep
            columns. If the proportion of NaN vallues is greater
            or equal to this threshold value, the column is removed.
            If set to ``1.0``, it removes all the columns that have
            only NaN values.
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
    >>> from grizz.transformer import DropNanColumn
    >>> transformer = DropNanColumn()
    >>> transformer
    DropNanColumnTransformer(columns=None, threshold=1.0, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
    ...         "col2": [1, None, 3, None, 5],
    ...         "col3": [None, None, None, None, None],
    ...     }
    ... )
    >>> frame
    shape: (5, 3)
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        threshold: float = 1.0,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns, exclude_columns=exclude_columns, missing_policy=missing_policy
        )
        self._threshold = threshold
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "threshold": self._threshold,
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
        logger.info(
            f"Checking columns and dropping the columns that have too "
            f"many NaN values (threshold={self._threshold})..."
        )
        if frame.is_empty():
            return frame
        columns = self.find_common_columns(frame)
        pct = frame.select((cs.numeric() & cs.by_name(columns)).is_nan()).sum() / frame.shape[0]
        cols = list(compress(pct.columns, (pct >= self._threshold).row(0)))
        logger.info(
            f"Dropping {len(cols):,} columns that have too "
            f"many NaN values (threshold={self._threshold})..."
        )
        logger.info(f"dropped columns: {cols}")
        out = frame.drop(cols)
        logger.info(str_shape_diff(orig=frame.shape, final=out.shape))
        return out


class DropNanRowTransformer(BaseInNTransformer):
    r"""Implement a transformer to drop all rows that contain NaN values.

    Note that all the values in the row need to be NaN to drop the
    row.

    Args:
        columns: The columns to check. If set to ``None`` (default),
            use all columns.
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

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import DropNanRow
    >>> transformer = DropNanRow()
    >>> transformer
    DropNanRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
    ...         "col2": [1.0, float("nan"), 3.0, float("nan"), float("nan")],
    ...         "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
    ...     }
    ... )
    >>> frame
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ f64  ┆ f64  │
    ╞══════╪══════╪══════╡
    │ 1.0  ┆ 1.0  ┆ NaN  │
    │ 2.0  ┆ NaN  ┆ NaN  │
    │ 3.0  ┆ 3.0  ┆ NaN  │
    │ 4.0  ┆ NaN  ┆ NaN  │
    │ NaN  ┆ NaN  ┆ NaN  │
    └──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (4, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ f64  ┆ f64  │
    ╞══════╪══════╪══════╡
    │ 1.0  ┆ 1.0  ┆ NaN  │
    │ 2.0  ┆ NaN  ┆ NaN  │
    │ 3.0  ┆ 3.0  ┆ NaN  │
    │ 4.0  ┆ NaN  ┆ NaN  │
    └──────┴──────┴──────┘

    ```
    """

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Dropping all rows that contain only NaN values in "
            f"{len(self.find_columns(frame)):,} columns...."
        )
        columns = self.find_common_columns(frame)
        out = frame.filter(~pl.all_horizontal((cs.numeric() & cs.by_name(columns)).is_nan()))
        logger.info(str_shape_diff(orig=frame.shape, final=out.shape))
        return out
