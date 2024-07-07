from __future__ import annotations

import pytest

from grizz.utils.format import human_byte, str_kwargs

################################
#     Tests for human_byte     #
################################


@pytest.mark.parametrize(
    ("size", "output"),
    [
        (2, "2.00 B"),
        (1023.0, "1,023.00 B"),
        (2048, "2.00 KB"),
        (2097152, "2.00 MB"),
        (2147483648, "2.00 GB"),
        (2199023255552, "2.00 TB"),
        (2251799813685248, "2.00 PB"),
        (2305843009213693952, "2,048.00 PB"),
    ],
)
def test_human_byte_decimal_2(size: int, output: str) -> None:
    assert human_byte(size) == output


@pytest.mark.parametrize(
    ("size", "output"),
    [
        (2, "2.000 B"),
        (1023.0, "1,023.000 B"),
        (2048, "2.000 KB"),
        (2097152, "2.000 MB"),
        (2147483648, "2.000 GB"),
        (2199023255552, "2.000 TB"),
        (2251799813685248, "2.000 PB"),
        (2305843009213693952, "2,048.000 PB"),
    ],
)
def test_human_byte_decimal_3(size: int, output: str) -> None:
    assert human_byte(size, decimal=3) == output


################################
#     Tests for str_kwargs     #
################################


def test_str_kwargs_0() -> None:
    assert str_kwargs({}) == ""


def test_str_kwargs_1() -> None:
    assert str_kwargs({"key1": 1}) == ", key1=1"


def test_str_kwargs_2() -> None:
    assert str_kwargs({"key1": 1, "key2": 2}) == ", key1=1, key2=2"
