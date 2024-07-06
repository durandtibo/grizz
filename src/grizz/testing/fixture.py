r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = ["clickhouse_connect_available", "tqdm_available"]

import pytest

from grizz.utils.imports import is_clickhouse_connect_available, is_tqdm_available

clickhouse_connect_available = pytest.mark.skipif(
    not is_clickhouse_connect_available(), reason="Requires clickhouse_connect"
)
tqdm_available = pytest.mark.skipif(not is_tqdm_available(), reason="Requires tqdm")
