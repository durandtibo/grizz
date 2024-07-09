r"""Contain transformers to transform columns or rows with null
values."""

from __future__ import annotations

__all__ = ["NullColumnTransformer"]

import logging
from itertools import compress
from typing import TYPE_CHECKING, Any

from grizz.transformer.columns import BaseColumnsTransformer
from grizz.utils.format import str_kwargs

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


logger = logging.getLogger(__name__)


class NullColumnTransformer(BaseColumnsTransformer):
    r"""Implement a transformer to remove the columns that have too many
    null values.

    Args:
        columns: The columns to convert. ``None`` means all the
            columns.
        threshold: The maximum percentage of null values to keep
            columns. If the proportion of null vallues is greater
            or equal to this threshold value, the column is removed.
            If set to ``1.0``, it removes all the columns that have
            only null values.
        dtype: The target data type.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.
        **kwargs: The keyword arguments for ``cast``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import NullColumn
    >>> transformer = NullColumn()
    >>> transformer
    NullColumnTransformer(columns=None, threshold=1.0, ignore_missing=False)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
    ...         "col2": [1, None, 3, None, 5],
    ...         "col3": [None, None, None, None, None],
    ...     }
    ... )
    >>> frame
    shape: (5, 3)
    ┌────────────┬──────┬──────┐
    │ col1       ┆ col2 ┆ col3 │
    │ ---        ┆ ---  ┆ ---  │
    │ str        ┆ i64  ┆ null │
    ╞════════════╪══════╪══════╡
    │ 2020-1-1   ┆ 1    ┆ null │
    │ 2020-1-2   ┆ null ┆ null │
    │ 2020-1-31  ┆ 3    ┆ null │
    │ 2020-12-31 ┆ null ┆ null │
    │ null       ┆ 5    ┆ null │
    └────────────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌────────────┬──────┐
    │ col1       ┆ col2 │
    │ ---        ┆ ---  │
    │ str        ┆ i64  │
    ╞════════════╪══════╡
    │ 2020-1-1   ┆ 1    │
    │ 2020-1-2   ┆ null │
    │ 2020-1-31  ┆ 3    │
    │ 2020-12-31 ┆ null │
    │ null       ┆ 5    │
    └────────────┴──────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        threshold: float = 1.0,
        ignore_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(columns, ignore_missing)
        self._threshold = threshold
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(columns={self._columns}, threshold={self._threshold}, "
            f"ignore_missing={self._ignore_missing}{str_kwargs(self._kwargs)})"
        )

    def _pre_transform(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"removing columns with too many missing values (threshold={self._threshold})..."
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.shape[0] == 0:
            return frame
        columns = self.find_common_columns(frame)
        orig_shape = len(columns)
        pct = frame.select(columns).null_count() / frame.shape[0]
        cols = list(compress(pct.columns, (pct >= self._threshold).row(0)))
        logger.info(
            f"Removing {len(cols):,} columns because they have too "
            f"many null values (threshold={self._threshold})..."
        )
        out = frame.drop(cols)
        logger.info(f"shape: {orig_shape} -> {out.shape}")
        return out
