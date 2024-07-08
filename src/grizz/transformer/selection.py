r"""Contain ``polars.DataFrame`` transformers to select columns in
DataFrames."""

from __future__ import annotations

__all__ = ["ColumnSelectionTransformer"]

import logging
from typing import TYPE_CHECKING

from grizz.transformer.base import BaseTransformer
from grizz.utils.column import find_common_columns, find_missing_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


logger = logging.getLogger(__name__)


class ColumnSelectionTransformer(BaseTransformer):
    r"""Implement a ``polars.DataFrame`` transformer to select a subset
    of columns.

    Args:
        columns: The columns to keep.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise a warning message is shown.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ColumnSelection
    >>> transformer = ColumnSelection(columns=["col1", "col2"])
    >>> transformer
    ColumnSelectionTransformer(columns=['col1', 'col2'], ignore_missing=False)
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

    def __init__(self, columns: Sequence[str], ignore_missing: bool = False) -> None:
        self._columns = list(columns)
        self._ignore_missing = bool(ignore_missing)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(columns={self._columns}, "
            f"ignore_missing={self._ignore_missing})"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"selecting {len(self._columns):,} columns: {self._columns}")
        missing = find_missing_columns(frame, self._columns)
        if missing and not self._ignore_missing:
            msg = f"{len(missing)} columns are missing in the DataFrame: {missing}"
            raise RuntimeError(msg)
        if missing:
            logger.warning(
                f"{len(missing)} columns are missing in the DataFrame and will be ignored: "
                f"{missing}"
            )

        columns = find_common_columns(frame, self._columns)
        out = frame.select(columns)
        logger.info(f"DataFrame shape after the column selection: {out.shape}")
        return out
