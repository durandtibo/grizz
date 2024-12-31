from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_allclose

from grizz.utils.sorting import mixed_typed_sort

if TYPE_CHECKING:
    from collections.abc import Iterable

######################################
#     Tests for mixed_typed_sort     #
######################################


@pytest.mark.parametrize(
    ("data", "output"),
    [
        # NaNs
        (
            [4, float("nan"), 2, 1.2, 7.9, -2, float("nan"), float("inf"), float("-inf")],
            [float("nan"), float("nan"), float("-inf"), -2, 1.2, 2, 4, 7.9, float("inf")],
        ),
        # numeric types
        ([2, 0.2, False, 8, 4.2, True], [False, 0.2, True, 2, 4.2, 8]),
        # int and string
        ([1, "c", "a", "b", 4, -2], [-2, 1, 4, "a", "b", "c"]),
    ],
)
def test_mixed_typed_sort(data: Iterable, output: Iterable) -> None:
    assert objects_are_allclose(mixed_typed_sort(data), output, equal_nan=True)


@pytest.mark.parametrize(
    ("data", "output"),
    [
        # NaNs
        (
            [4, float("nan"), 2, 1.2, 7.9, -2, float("nan"), float("inf"), float("-inf")],
            [float("inf"), 7.9, 4, 2, 1.2, -2, float("-inf"), float("nan"), float("nan")],
        ),
        # numeric types
        ([2, 0.2, False, 8, 4.2, True], [8, 4.2, 2, True, 0.2, False]),
        # int and string
        ([1, "c", "a", "b", 4, -2], [4, 1, -2, "c", "b", "a"]),
    ],
)
def test_mixed_typed_sort_reverse_true(data: Iterable, output: Iterable) -> None:
    assert objects_are_allclose(mixed_typed_sort(data, reverse=True), output, equal_nan=True)
