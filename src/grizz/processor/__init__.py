r"""Contain ``polars.DataFrame`` processors."""

from __future__ import annotations

__all__ = [
    "BaseProcessor",
    "is_processor_config",
    "setup_processor",
]

from grizz.processor.base import BaseProcessor, is_processor_config, setup_processor
