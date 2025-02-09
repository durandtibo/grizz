r"""Contain transformers to drop columns or rows with NaN values."""

from __future__ import annotations

__all__ = ["DropNanRowTransformer"]

import logging

import polars as pl
import polars.selectors as cs

from grizz.lazy.transformer.columns import BaseInNTransformer
from grizz.transformer.utils import get_classname, message_skip_fit

logger = logging.getLogger(__name__)


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
    >>> from grizz.lazy.transformer import DropNanRow
    >>> transformer = DropNanRow()
    >>> transformer
    DropNanRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.LazyFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
    ...         "col2": [1.0, float("nan"), 3.0, float("nan"), float("nan")],
    ...         "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
    ...     }
    ... )
    >>> frame.collect()
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
    >>> out.collect()
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

    def _fit(self, frame: pl.LazyFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        columns = self.find_common_columns(frame)
        logger.info(
            f"Dropping all rows that contain only NaN values in {len(columns):,} columns...."
        )
        return frame.filter(~pl.all_horizontal((cs.float() & cs.by_name(columns)).is_nan()))
