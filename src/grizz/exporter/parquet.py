r"""Contain the implementation of a parquet DataFrame exporter."""

from __future__ import annotations

__all__ = ["ParquetExporter"]

import logging
from typing import TYPE_CHECKING, Any

from grizz.exporter.base import BaseExporter
from grizz.utils.format import str_kwargs
from grizz.utils.path import human_file_size, sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

logger = logging.getLogger(__name__)


class ParquetExporter(BaseExporter):
    r"""Implement a parquet DataFrame exporter.

    Args:
        path: The path to the parquet file to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.DataFrame.write_parquet``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.exporter import ParquetExporter
    >>> exporter = ParquetExporter(path="/path/to/frame.parquet")
    >>> exporter
    ParquetExporter(path=/path/to/frame.parquet)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> exporter.export(frame)  # doctest: +SKIP

    ```
    """

    def __init__(self, path: Path | str, **kwargs: Any) -> None:
        self._path = sanitize_path(path)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(path={self._path}{str_kwargs(self._kwargs)})"

    def export(self, frame: pl.DataFrame) -> None:
        logger.info(
            f"Exporting the DataFrame of shape={frame.shape} to parquet file {self._path} ..."
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(self._path, **self._kwargs)
        logger.info(f"DataFrame exported | size={human_file_size(self._path)}")
