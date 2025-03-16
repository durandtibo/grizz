r"""Contain the implementation of CSV ingestors."""

from __future__ import annotations

__all__ = ["CsvFileIngestor", "CsvIngestor"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from coola import objects_are_equal

from grizz.ingestor.utils import check_data_file
from grizz.lazy.ingestor.base import BaseIngestor
from grizz.utils.format import str_kwargs
from grizz.utils.path import human_file_size, sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

    from grizz.ingestor.parquet import FileSource

logger = logging.getLogger(__name__)


class CsvIngestor(BaseIngestor):
    r"""Implement a CSV ingestor.

    Args:
        source: The source to the CSV data to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.scan_csv``.

    Example usage:

    ```pycon

    >>> from grizz.lazy.ingestor import CsvIngestor
    >>> ingestor = CsvIngestor(source="/path/to/frame.csv")
    >>> ingestor
    CsvIngestor(source=/path/to/frame.csv)
    >>> frame = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(self, source: FileSource, **kwargs: Any) -> None:
        self._source = source
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(source={self._source}{str_kwargs(self._kwargs)})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._source == other._source and objects_are_equal(
            self._kwargs, other._kwargs, equal_nan=equal_nan
        )

    def ingest(self) -> pl.LazyFrame:
        logger.info(f"Ingesting CSV data from {self._source}...")
        frame = pl.scan_csv(self._source, **self._kwargs)
        logger.info(f"LazyFrame ingested | schema={frame.collect_schema()}")
        return frame


class CsvFileIngestor(CsvIngestor):
    r"""Implement a CSV file ingestor.

    Args:
        path: The path to the CSV file to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.scan_csv``.

    Example usage:

    ```pycon

    >>> from grizz.lazy.ingestor import CsvFileIngestor
    >>> ingestor = CsvFileIngestor(path="/path/to/frame.csv")
    >>> ingestor
    CsvFileIngestor(source=/path/to/frame.csv)
    >>> frame = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(self, path: Path | str, **kwargs: Any) -> None:
        super().__init__(source=sanitize_path(path), **kwargs)

    def ingest(self) -> pl.LazyFrame:
        check_data_file(self._source)
        logger.info(f"Ingesting CSV file {self._source} | size={human_file_size(self._source)}")
        return super().ingest()
