r"""Contain the implementation of a CSV LazyFrame exporter."""

from __future__ import annotations

__all__ = ["CsvExporter"]

import logging
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from grizz.lazy.exporter.base import BaseExporter
from grizz.utils.format import str_kwargs
from grizz.utils.path import human_file_size, sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

logger = logging.getLogger(__name__)


class CsvExporter(BaseExporter):
    r"""Implement a CSV LazyFrame exporter.

    Args:
        path: The path to the csv file to ingest.
        **kwargs: Additional keyword arguments for
            ``polars.LazyFrame.write_csv``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.lazy.exporter import CsvExporter
    >>> exporter = CsvExporter(path="/path/to/frame.csv")
    >>> exporter
    CsvExporter(path=/path/to/frame.csv)
    >>> frame = pl.LazyFrame(
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

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._path == other._path and objects_are_equal(
            self._kwargs, other._kwargs, equal_nan=equal_nan
        )

    def export(self, frame: pl.LazyFrame) -> None:
        logger.info(f"Exporting the LazyFrame  to CSV file {self._path} ...")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        frame.sink_csv(self._path, **self._kwargs)
        logger.info(f"LazyFrame exported | size={human_file_size(self._path)}")
