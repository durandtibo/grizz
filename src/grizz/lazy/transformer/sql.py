r"""Contain a transformer that executes a SQL query against the
LazyFrame."""

from __future__ import annotations

__all__ = ["SqlTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from grizz.lazy.transformer.columns import BaseArgTransformer

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class SqlTransformer(BaseArgTransformer):
    r"""Implement a transformer that executes a SQL query against the
    LazyFrame.

    Args:
        query: The SQL query to execute.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.lazy.transformer import SqlTransformer
    >>> transformer = SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")
    >>> transformer
    SqlTransformer(
      (query): SELECT col1, col4 FROM self WHERE col1 > 2
    )
    >>> frame = pl.LazyFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> out = transformer.transform(frame)
    >>> out.collect()
    shape: (3, 2)
    ┌──────┬──────┐
    │ col1 ┆ col4 │
    │ ---  ┆ ---  │
    │ i64  ┆ str  │
    ╞══════╪══════╡
    │ 3    ┆ c    │
    │ 4    ┆ d    │
    │ 5    ┆ e    │
    └──────┴──────┘

    ```
    """

    def __init__(self, query: str) -> None:
        self._query = query

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"query": self._query}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def get_args(self) -> dict:
        return {"query": self._query}

    def fit(self, frame: pl.LazyFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        logger.info(f"Executing the following SQL query:\n{self._query}")
        return frame.sql(self._query)
