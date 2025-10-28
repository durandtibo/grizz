from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from grizz.utils.series import compute_stats_boolean

###########################################
#     Tests for compute_stats_boolean     #
###########################################


def test_compute_stats_boolean() -> None:
    assert objects_are_equal(
        compute_stats_boolean(pl.Series([True, False, None, None, False, None])),
        {"num_false": 2, "num_null": 3, "num_true": 1, "total": 6},
    )


def test_compute_stats_boolean_empty() -> None:
    assert objects_are_equal(
        compute_stats_boolean(pl.Series(dtype=pl.Boolean)),
        {"num_false": 0, "num_null": 0, "num_true": 0, "total": 0},
    )


def test_compute_stats_boolean_incorrect_dtype() -> None:
    with pytest.raises(ValueError, match=r"Incorrect dtype"):
        compute_stats_boolean(pl.Series())
