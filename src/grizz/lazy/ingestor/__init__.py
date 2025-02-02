r"""Contain LazyFrame ingestors."""

from __future__ import annotations

__all__ = ["BaseIngestor", "Ingestor", "is_ingestor_config", "setup_ingestor"]

from grizz.lazy.ingestor.base import BaseIngestor, is_ingestor_config, setup_ingestor
from grizz.lazy.ingestor.vanilla import Ingestor
