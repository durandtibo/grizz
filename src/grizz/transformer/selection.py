r"""Contain ``polars.DataFrame`` transformers to select columns in
DataFrames."""

from __future__ import annotations

__all__ = ["ColumnSelectionTransformer"]

import logging
from typing import TYPE_CHECKING

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_shape_diff

if TYPE_CHECKING:

    import polars as pl


logger = logging.getLogger(__name__)


class ColumnSelectionTransformer(BaseInNTransformer):
    r"""Implement a ``polars.DataFrame`` transformer to select a subset
    of columns.

    Args:
        columns: The columns to keep.
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
    >>> from grizz.transformer import ColumnSelection
    >>> transformer = ColumnSelection(columns=["col1", "col2"])
    >>> transformer
    ColumnSelectionTransformer(columns=('col1', 'col2'), exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
    ...         "col2": [1, 2, 3, 4, 5],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> out = transformer.transform(frame)
    >>> out
        shape: (5, 2)
    ┌────────────┬──────┐
    │ col1       ┆ col2 │
    │ ---        ┆ ---  │
    │ str        ┆ i64  │
    ╞════════════╪══════╡
    │ 2020-1-1   ┆ 1    │
    │ 2020-1-2   ┆ 2    │
    │ 2020-1-31  ┆ 3    │
    │ 2020-12-31 ┆ 4    │
    │ null       ┆ 5    │
    └────────────┴──────┘

    ```
    """

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Selecting {len(self.find_columns(frame)):,} columns...")
        columns = self.find_common_columns(frame)
        initial_shape = frame.shape
        out = frame.select(columns)
        logger.info(str_shape_diff(orig=initial_shape, final=out.shape))
        return out
