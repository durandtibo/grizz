r"""Contain data ingestors."""

from __future__ import annotations

__all__ = [
    "BaseIngestor",
    "Ingestor",
    "is_ingestor_config",
    "setup_ingestor",
]

from grizz.ingestor.base import BaseIngestor, is_ingestor_config, setup_ingestor
from grizz.ingestor.vanilla import Ingestor
