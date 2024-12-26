from __future__ import annotations

import pytest

from grizz.utils.format import (
    human_byte,
    str_col_diff,
    str_dataframe_shape_diff,
    str_kwargs,
    str_row_diff,
)

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


##################################
#     Tests for str_col_diff     #
##################################


def test_str_col_diff_zero() -> None:
    assert str_col_diff(0, 0) == "0/0 (nan %) column has been removed"


def test_str_col_diff_one() -> None:
    assert str_col_diff(100, 99) == "1/100 (1.0000 %) column has been removed"


def test_str_col_diff_multiple() -> None:
    assert str_col_diff(100, 10) == "90/100 (90.0000 %) columns have been removed"


##################################
#     Tests for str_row_diff     #
##################################


def test_str_row_diff_zero() -> None:
    assert str_row_diff(0, 0) == "0/0 (nan %) row has been removed"


def test_str_row_diff_one() -> None:
    assert str_row_diff(100, 99) == "1/100 (1.0000 %) row has been removed"


def test_str_row_diff_multiple() -> None:
    assert str_row_diff(100, 10) == "90/100 (90.0000 %) rows have been removed"


##############################################
#     Tests for str_dataframe_shape_diff     #
##############################################


def test_str_dataframe_shape_diff_zero() -> None:
    assert (
        str_dataframe_shape_diff(orig=(0, 0), final=(0, 0)) == "DataFrame shape: (0, 0) -> (0, 0)"
    )


def test_str_dataframe_shape_diff_same_shape() -> None:
    assert (
        str_dataframe_shape_diff(orig=(100, 5), final=(100, 5))
        == "DataFrame shape: (100, 5) -> (100, 5)"
    )


def test_str_dataframe_shape_diff_cols() -> None:
    assert (
        str_dataframe_shape_diff(orig=(100, 5), final=(100, 3))
        == "DataFrame shape: (100, 5) -> (100, 3) | 2/5 (40.0000 %) columns have been removed"
    )


def test_str_dataframe_shape_diff_rows() -> None:
    assert (
        str_dataframe_shape_diff(orig=(100, 5), final=(80, 5))
        == "DataFrame shape: (100, 5) -> (80, 5) | 20/100 (20.0000 %) rows have been removed"
    )


def test_str_dataframe_shape_diff_cols_and_rows() -> None:
    assert (
        str_dataframe_shape_diff(orig=(100, 10), final=(80, 8))
        == "DataFrame shape: (100, 10) -> (80, 8) | 20/100 (20.0000 %) rows have been removed | "
        "2/10 (20.0000 %) columns have been removed"
    )
