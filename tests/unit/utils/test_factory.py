from __future__ import annotations

from collections import deque
from typing import Any

import pytest
from objectory import OBJECT_TARGET

from grizz.utils.factory import setup_object

##################################
#     Tests for setup_object     #
##################################


@pytest.mark.parametrize(
    "module", [deque(), {OBJECT_TARGET: "collections.deque", "iterable": [1, 2, 1, 3]}]
)
def test_setup_object(module: Any) -> None:
    assert isinstance(setup_object(module), deque)


def test_setup_object_object() -> None:
    obj = deque([1, 2, 1, 3])
    assert setup_object(obj) is obj
