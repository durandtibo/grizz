from __future__ import annotations

import pytest

from grizz.utils.hashing import str_to_sha256

###################################
#     Tests for str_to_sha256     #
###################################


@pytest.mark.parametrize("value", ["meow", 1, "123", "abcdefghijklmnopqrstuvwxyz"])
def test_str_to_sha256(value: str) -> None:
    out = str_to_sha256(value)
    assert isinstance(out, str)
    assert out != value
