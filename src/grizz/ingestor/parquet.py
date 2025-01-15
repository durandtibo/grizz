r"""Contain the implementation of parquet ingestors."""

from __future__ import annotations

__all__ = ["ParquetFileIngestor", "ParquetIngestor"]

import logging
from pathlib import Path
from typing import IO, Any, Union

import polars as pl
from coola import objects_are_equal
from iden.utils.time import timeblock

from grizz.ingestor.base import BaseIngestor
from grizz.ingestor.utils import check_dataframe_file
from grizz.utils.format import human_byte, str_kwargs
from grizz.utils.path import human_file_size, sanitize_path

FileSource = Union[
    str,
    Path,
    IO[bytes],
    bytes,
    list[str],
    list[Path],
    list[IO[bytes]],
    list[bytes],
]

logger = logging.getLogger(__name__)


class ParquetIngestor(BaseIngestor):
    r"""Implement a parquet ingestor.

    Args:
        source: The source to the parquet data to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.read_parquet``.

    Example usage:

    ```pycon

    >>> from grizz.ingestor import ParquetIngestor
    >>> ingestor = ParquetIngestor(source="/path/to/frame.parquet")
    >>> ingestor
    ParquetIngestor(source=/path/to/frame.parquet)
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

    def ingest(self) -> pl.DataFrame:
        logger.info(f"Ingesting parquet data from {self._source}...")
        with timeblock("DataFrame ingestion time: {time}"):
            frame = pl.read_parquet(self._source, **self._kwargs)
            logger.info(
                f"DataFrame ingested | shape={frame.shape}  "
                f"estimated size={human_byte(frame.estimated_size())}"
            )
        return frame


class ParquetFileIngestor(ParquetIngestor):
    r"""Implement a parquet file ingestor.

    Args:
        path: The path to the parquet file to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.read_parquet``.

    Example usage:

    ```pycon

    >>> from grizz.ingestor import ParquetFileIngestor
    >>> ingestor = ParquetFileIngestor(path="/path/to/frame.parquet")
    >>> ingestor
    ParquetFileIngestor(source=/path/to/frame.parquet)
    >>> frame = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(self, path: Path | str, **kwargs: Any) -> None:
        super().__init__(source=sanitize_path(path), **kwargs)

    def ingest(self) -> pl.DataFrame:
        check_dataframe_file(self._source)
        logger.info(f"Ingesting parquet file {self._source} | size={human_file_size(self._source)}")
        return super().ingest()
