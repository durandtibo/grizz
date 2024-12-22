r"""Contain ``polars.DataFrame`` transformers to copy columns."""

from __future__ import annotations

__all__ = ["CopyColumnsTransformer"]

import logging
from typing import TYPE_CHECKING

import polars as pl
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseColumnsTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class CopyColumnsTransformer(BaseColumnsTransformer):
    r"""Implement a transformer to copy some columns.

    Args:
        columns: The columns to copy. ``None`` means all the
            columns.
        prefix: The column name prefix for the copied columns.
        suffix: The column name suffix for the copied columns.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import CopyColumns
    >>> transformer = CopyColumns(columns=["col1", "col3"], prefix="", suffix="_raw")
    >>> transformer
    CopyColumnsTransformer(columns=('col1', 'col3'), prefix='', suffix='_raw', ignore_missing=False)
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
    shape: (5, 6)
    ┌──────┬──────┬──────┬──────┬──────────┬──────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_raw ┆ col3_raw │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---      ┆ ---      │
    │ i64  ┆ str  ┆ str  ┆ str  ┆ i64      ┆ str      │
    ╞══════╪══════╪══════╪══════╪══════════╪══════════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    ┆ 1        ┆ 1        │
    │ 2    ┆ 2    ┆ 2    ┆ b    ┆ 2        ┆ 2        │
    │ 3    ┆ 3    ┆ 3    ┆ c    ┆ 3        ┆ 3        │
    │ 4    ┆ 4    ┆ 4    ┆ d    ┆ 4        ┆ 4        │
    │ 5    ┆ 5    ┆ 5    ┆ e    ┆ 5        ┆ 5        │
    └──────┴──────┴──────┴──────┴──────────┴──────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        ignore_missing: bool = False,
    ) -> None:
        super().__init__(columns, ignore_missing)
        self._prefix = prefix
        self._suffix = suffix

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "prefix": self._prefix,
                "suffix": self._suffix,
                "ignore_missing": self._ignore_missing,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def _pre_fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _fit(self, frame: pl.DataFrame) -> None:
        pass  # no parameter to fit for this transformer.

    def _pre_transform(self, frame: pl.DataFrame) -> None:
        columns = self.find_columns(frame)
        logger.info(
            f"Copying {len(columns):,} columns | prefix={self._prefix!r} | "
            f"suffix={self._suffix!r} ..."
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        columns = self.find_common_columns(frame)
        return frame.with_columns(
            frame.select(pl.col(columns)).rename(lambda name: f"{self._prefix}{name}{self._suffix}")
        )
