r"""Contain the implementation of an ingestor that attempts the fast
ingestor first, falling back to the slow ingestor if needed."""

from __future__ import annotations

__all__ = ["CacheIngestor"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from grizz.exceptions import DataFrameNotFoundError
from grizz.exporter.base import BaseExporter, setup_exporter
from grizz.ingestor.base import BaseIngestor, setup_ingestor

if TYPE_CHECKING:

    import polars as pl

logger = logging.getLogger(__name__)


class CacheIngestor(BaseIngestor):
    r"""Implement an ingestor that attempts the fast ingestor first,
    falling back to the slow ingestor if needed.

    Internally, this ingestor attempts to load the DataFrame using
    the fast ingestor. If a DataFrameNotFoundError is raised,
    it falls back to the slow ingestor, then exports the DataFrame
    for ingestion by the fast ingestor during the next cycle.

    Args:
        ingestor_slow: The slow DataFrame ingestor or its
            configuration.
        ingestor_fast: The fast DataFrame ingestor or its
            configuration.
        exporter: The DataFrame exporter or its configuration.
            The DataFrame exporter is responsible for storing the
            output of the slower DataFrame ingestor, allowing it to
            be ingested by the faster DataFrame ingestor during the
            next ingestion cycle.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.ingestor import CacheIngestor, Ingestor
    >>> from grizz.exporter import InMemoryExporter
    >>> ingestor_slow = Ingestor(
    ...     pl.DataFrame(
    ...         {
    ...             "col1": ["1", "2", "3", "4", "5"],
    ...             "col2": ["a", "b", "c", "d", "e"],
    ...             "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
    ...         }
    ...     )
    ... )
    >>> exporter_ingestor = InMemoryExporter()
    >>> ingestor = CacheIngestor(
    ...     ingestor_slow=ingestor_slow,
    ...     ingestor_fast=exporter_ingestor,
    ...     exporter=exporter_ingestor,
    ... )
    >>> ingestor
    CacheIngestor(
      (ingestor_slow): Ingestor(shape=(5, 3))
      (ingestor_fast): InMemoryExporter(frame=None)
      (exporter): InMemoryExporter(frame=None)
    )
    >>> frame = ingestor.ingest()
    >>> frame
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ str  ┆ str  ┆ f64  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ a    ┆ 1.2  │
    │ 2    ┆ b    ┆ 2.2  │
    │ 3    ┆ c    ┆ 3.2  │
    │ 4    ┆ d    ┆ 4.2  │
    │ 5    ┆ e    ┆ 5.2  │
    └──────┴──────┴──────┘

    ```
    """

    def __init__(
        self,
        ingestor_slow: BaseIngestor | dict,
        ingestor_fast: BaseIngestor | dict,
        exporter: BaseExporter | dict,
    ) -> None:
        self._ingestor_slow = setup_ingestor(ingestor_slow)
        self._ingestor_fast = setup_ingestor(ingestor_fast)
        self._exporter = setup_exporter(exporter)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "ingestor_slow": self._ingestor_slow,
                    "ingestor_fast": self._ingestor_fast,
                    "exporter": self._exporter,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def ingest(self) -> pl.DataFrame:
        try:
            frame = self._ingestor_fast.ingest()
        except DataFrameNotFoundError:
            frame = self._ingestor_slow.ingest()
            self._exporter.export(frame)
        return frame
