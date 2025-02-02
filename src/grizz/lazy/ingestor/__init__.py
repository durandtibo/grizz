r"""Contain LazyFrame ingestors."""

from __future__ import annotations

__all__ = [
    "BaseIngestor",
    "CsvFileIngestor",
    "CsvIngestor",
    "Ingestor",
    "JoinIngestor",
    "ParquetFileIngestor",
    "ParquetIngestor",
    "is_ingestor_config",
    "setup_ingestor",
]

from grizz.lazy.ingestor.base import BaseIngestor, is_ingestor_config, setup_ingestor
from grizz.lazy.ingestor.csv import CsvFileIngestor, CsvIngestor
from grizz.lazy.ingestor.join import JoinIngestor
from grizz.lazy.ingestor.parquet import ParquetFileIngestor, ParquetIngestor
from grizz.lazy.ingestor.vanilla import Ingestor
