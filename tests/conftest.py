from __future__ import annotations

__all__ = ["polars_greater_equal_1_17"]

import operator

import pytest
from feu import compare_version

SKLEARN_GREATER_EQUAL_1_17 = compare_version("polars", operator.ge, "1.17.0")

polars_greater_equal_1_17 = pytest.mark.skipif(
    not SKLEARN_GREATER_EQUAL_1_17, reason="Requires polars>=1.17.0"
)
