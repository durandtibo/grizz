r"""Contain the implementation of a in-memory DataFrame exporter and
ingestor."""

from __future__ import annotations

__all__ = ["InMemoryExporter"]

import logging
from typing import TYPE_CHECKING

from grizz.exceptions import DataFrameNotFoundError
from grizz.exporter.base import BaseExporter
from grizz.ingestor.base import BaseIngestor

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class InMemoryExporter(BaseExporter, BaseIngestor):
    r"""Implement an in-memory DataFrame exporter and ingestor.

    Notes:
        This exporter is both exporter and ingestor as the object
            stores the DataFrame.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.exporter import InMemoryExporter
    >>> exporter = InMemoryExporter()
    >>> exporter
    InMemoryExporter(frame=None)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> exporter.export(frame)
    >>> df = exporter.ingest()
    >>> df
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┘

    ```
    """

    def __init__(self) -> None:
        self._frame = None

    def __repr__(self) -> str:
        frame = None if self._frame is None else self._frame.shape
        return f"{self.__class__.__qualname__}(frame={frame})"

    def export(self, frame: pl.DataFrame) -> None:
        logger.info(f"Exporting the DataFrame (shape={frame.shape}) to memory ...")
        self._frame = frame

    def ingest(self) -> pl.DataFrame:
        logger.info("Ingesting DataFrame...")
        if self._frame is None:
            msg = (
                "No DataFrame available for ingestion. You must export a DataFrame "
                "before to ingest it"
            )
            raise DataFrameNotFoundError(msg)
        return self._frame
