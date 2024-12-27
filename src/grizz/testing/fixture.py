r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = [
    "clickhouse_connect_available",
    "colorlog_available",
    "pyarrow_available",
    "sklearn_available",
    "tqdm_available",
]

import pytest

from grizz.utils.imports import (
    is_clickhouse_connect_available,
    is_colorlog_available,
    is_pyarrow_available,
    is_sklearn_available,
    is_tqdm_available,
)

clickhouse_connect_available = pytest.mark.skipif(
    not is_clickhouse_connect_available(), reason="requires clickhouse_connect"
)
colorlog_available = pytest.mark.skipif(not is_colorlog_available(), reason="requires colorlog")
pyarrow_available = pytest.mark.skipif(not is_pyarrow_available(), reason="requires pyarrow")
sklearn_available = pytest.mark.skipif(not is_sklearn_available(), reason="requires sklearn")
tqdm_available = pytest.mark.skipif(not is_tqdm_available(), reason="requires tqdm")
