r"""Contain the implementation of a parquet ingestor."""

from __future__ import annotations

__all__ = ["ParquetIngestor"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from iden.utils.time import timeblock

from grizz.ingestor.base import BaseIngestor
from grizz.ingestor.utils import check_dataframe_file
from grizz.utils.format import human_byte, str_kwargs
from grizz.utils.path import human_file_size, sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ParquetIngestor(BaseIngestor):
    r"""Implement a parquet DataFrame ingestor.

    Args:
        path: The path to the parquet file to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.read_parquet``.

    Example usage:

    ```pycon

    >>> from grizz.ingestor import ParquetIngestor
    >>> ingestor = ParquetIngestor(path="/path/to/frame.parquet")
    >>> ingestor
    ParquetIngestor(path=/path/to/frame.parquet)
    >>> frame = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(self, path: Path | str, **kwargs: Any) -> None:
        self._path = sanitize_path(path)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(path={self._path}{str_kwargs(self._kwargs)})"

    def ingest(self) -> pl.DataFrame:
        check_dataframe_file(self._path)
        logger.info(
            f"Ingesting parquet data from {self._path} | size={human_file_size(self._path)}..."
        )
        with timeblock("DataFrame ingestion time: {time}"):
            frame = pl.read_parquet(self._path, **self._kwargs)
            logger.info(
                f"DataFrame ingested | shape={frame.shape}  "
                f"estimated size={human_byte(frame.estimated_size())}"
            )
        return frame
