r"""Contain DataFrame ingestors."""

from __future__ import annotations

__all__ = [
    "BaseIngestor",
    "CacheIngestor",
    "ClickHouseIngestor",
    "CsvIngestor",
    "Ingestor",
    "ParquetIngestor",
    "TransformIngestor",
    "is_ingestor_config",
    "setup_ingestor",
]

from grizz.ingestor.base import BaseIngestor, is_ingestor_config, setup_ingestor
from grizz.ingestor.cache import CacheIngestor
from grizz.ingestor.clickhouse import ClickHouseIngestor
from grizz.ingestor.csv import CsvIngestor
from grizz.ingestor.parquet import ParquetIngestor
from grizz.ingestor.transform import TransformIngestor
from grizz.ingestor.vanilla import Ingestor
