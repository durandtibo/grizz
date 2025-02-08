r"""Contain ``polars.DataFrame`` transformers to select rows in
DataFrames."""

from __future__ import annotations

__all__ = ["FirstRowTransformer"]

import logging
from typing import TYPE_CHECKING

from grizz.transformer.columns import BaseArgTransformer
from grizz.transformer.utils import get_classname, message_skip_fit

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class FirstRowTransformer(BaseArgTransformer):
    r"""Implement a transformer that select the first ``n`` rows.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import FirstRow
    >>> transformer = FirstRow(n=3)
    >>> transformer
    FirstRowTransformer(n=3)
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
    shape: (3, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def get_args(self) -> dict:
        return {"n": self._n}

    def _fit_dataframe(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform_dataframe(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Select {self._n:,} rows...")
        return frame.limit(self._n)
