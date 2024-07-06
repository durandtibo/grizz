r"""Contain data ingestors."""

from __future__ import annotations

__all__ = [
    "BaseIngestor",
    "ClickHouseIngestor",
    "CsvIngestor",
    "Ingestor",
    "ParquetIngestor",
    "is_ingestor_config",
    "setup_ingestor",
]

from grizz.ingestor.base import BaseIngestor, is_ingestor_config, setup_ingestor
from grizz.ingestor.clickhouse import ClickHouseIngestor
from grizz.ingestor.csv import CsvIngestor
from grizz.ingestor.parquet import ParquetIngestor
from grizz.ingestor.vanilla import Ingestor
