r"""Define some utility functions/classes."""

from __future__ import annotations

__all__ = ["ExamplePair"]

from dataclasses import dataclass
from typing import Any

from coola import objects_are_allclose, objects_are_equal

COMPARATOR_FUNCTIONS = [objects_are_equal, objects_are_allclose]


@dataclass
class ExamplePair:
    actual: Any
    expected: Any
    expected_message: str | None = None
    atol: float = 0.0
    rtol: float = 0.0
