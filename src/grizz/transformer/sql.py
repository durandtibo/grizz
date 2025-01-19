r"""Contain a transformer that executes a SQL query against the
DataFrame."""

from __future__ import annotations

__all__ = ["SqlTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from grizz.transformer.columns import BaseArgTransformer

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class SqlTransformer(BaseArgTransformer):
    r"""Implement a transformer that executes a SQL query against the
    DataFrame.

    Args:
        query: The SQL query to execute.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import SqlTransformer
    >>> transformer = SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")
    >>> transformer
    SqlTransformer(
      (query): SELECT col1, col4 FROM self WHERE col1 > 2
    )
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> out = transformer.transform(frame)
    >>> out
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

    def _fit_dataframe(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform_dataframe(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Executing the following SQL query:\n{self._query}")
        return frame.sql(self._query)
