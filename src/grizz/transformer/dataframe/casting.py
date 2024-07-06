r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = ["CastDataFrameTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from grizz.transformer.dataframe.base import BaseDataFrameTransformer
from grizz.utils.imports import is_tqdm_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_tqdm_available():
    from tqdm import tqdm
else:  # pragma: no cover
    from grizz.utils.noop import tqdm

logger = logging.getLogger(__name__)


class CastDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to convert some columns to a new data
    type.

    Args:
        columns: The columns to convert.
        dtype: The target data type.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.
        **kwargs: The keyword arguments for ``cast``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer.dataframe import Cast
    >>> transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    >>> transformer
    CastDataFrameTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)
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
        columns: Sequence[str],
        dtype: type[pl.DataType],
        ignore_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        self._columns = tuple(columns)
        self._dtype = dtype
        self._ignore_missing = bool(ignore_missing)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join([f"{key}={value}" for key, value in self._kwargs.items()])
        if args:
            args = ", " + args
        return (
            f"{self.__class__.__qualname__}(columns={self._columns}, dtype={self._dtype}, "
            f"ignore_missing={self._ignore_missing}{args})"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for col in tqdm(self._columns, desc=f"converting to {self._dtype}"):
            if col not in frame:
                if self._ignore_missing:
                    logger.warning(
                        f"skipping transformation for column {col} because the column is missing"
                    )
                else:
                    msg = f"column {col} is not in the DataFrame (columns:{sorted(frame.columns)})"
                    raise RuntimeError(msg)
            else:
                logger.info(f"transforming column `{col}`...")
                frame = frame.with_columns(
                    frame.select(pl.col(col).cast(self._dtype, **self._kwargs))
                )
        return frame
