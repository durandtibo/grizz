r"""Contain transformers to drop columns or rows with null values."""

from __future__ import annotations

__all__ = ["DropDuplicateTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_kwargs, str_shape_diff

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class DropDuplicateTransformer(BaseInNTransformer):
    r"""Implement a transformer to drop duplicate rows.

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
        **kwargs: The keyword arguments for ``unique``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import DropDuplicate
    >>> transformer = DropDuplicate(keep="first", maintain_order=True)
    >>> transformer
    DropDuplicateTransformer(columns=None, exclude_columns=(), missing_policy='raise', keep=first, maintain_order=True)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 1],
    ...         "col2": ["1", "2", "3", "4", "1"],
    ...         "col3": ["1", "2", "3", "1", "1"],
    ...         "col4": ["a", "a", "a", "a", "a"],
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
    │ 2    ┆ 2    ┆ 2    ┆ a    │
    │ 3    ┆ 3    ┆ 3    ┆ a    │
    │ 4    ┆ 4    ┆ 1    ┆ a    │
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (4, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ a    │
    │ 3    ┆ 3    ┆ 3    ┆ a    │
    │ 4    ┆ 4    ┆ 1    ┆ a    │
    └──────┴──────┴──────┴──────┘

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
        logger.info(
            f"Dropping duplicate rows by checking {len(self.find_columns(frame)):,} columns...."
        )
        columns = self.find_common_columns(frame)
        initial_shape = frame.shape
        out = frame.unique(subset=cs.by_name(columns), **self._kwargs)
        logger.info(str_shape_diff(orig=initial_shape, final=out.shape))
        return out
