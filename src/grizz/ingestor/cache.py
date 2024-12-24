from __future__ import annotations

__all__ = ["CacheIngestor"]

import datetime
import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping
from iden.io import save_text

from grizz.exporter.base import BaseExporter, setup_exporter
from grizz.ingestor.base import BaseIngestor, setup_ingestor
from grizz.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

logger = logging.getLogger(__name__)


class CacheIngestor(BaseIngestor):
    r"""Implement an ingestor that also transforms the DataFrame.

    Args:
        ingestor_slow: The slow ingestor.
        ingestor_fast: The fast ingestor.
        exporter: The DataFrame exporter.

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
    ...     exporter=exporter_ingestor
    ... )
    >>> ingestor
    CacheIngestor(
    >>> frame = ingestor.ingest()
    >>> frame
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ f32  ┆ str  ┆ f32  │
    ╞══════╪══════╪══════╡
    │ 1.0  ┆ a    ┆ 1.2  │
    │ 2.0  ┆ b    ┆ 2.2  │
    │ 3.0  ┆ c    ┆ 3.2  │
    │ 4.0  ┆ d    ┆ 4.2  │
    │ 5.0  ┆ e    ┆ 5.2  │
    └──────┴──────┴──────┘

    ```
    """

    def __init__(
        self,
        ingestor_slow: BaseIngestor | dict,
        ingestor_fast: BaseIngestor | dict,
        exporter: BaseExporter | dict,
        path: Path | str,
    ) -> None:
        self._ingestor_slow = setup_ingestor(ingestor_slow)
        self._ingestor_fast = setup_ingestor(ingestor_fast)
        self._exporter = setup_exporter(exporter)
        self._path = sanitize_path(path)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "ingestor_slow": self._ingestor_slow,
                    "ingestor_fast": self._ingestor_fast,
                    "exporter": self._exporter,
                    "path": self._path,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def ingest(self) -> pl.DataFrame:
        if self._path.is_file():
            return self._ingestor_fast.ingest()
        frame = self._ingestor_slow.ingest()
        self._exporter.export(frame)
        save_text(
            f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            self._path,
        )
        return frame
