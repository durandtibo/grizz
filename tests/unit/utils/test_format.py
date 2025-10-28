from __future__ import annotations

import polars as pl
import pytest

from grizz.utils.format import (
    human_byte,
    str_boolean_series_stats,
    str_col_diff,
    str_dataframe_diff,
    str_kwargs,
    str_row_diff,
    str_shape_diff,
    str_size_diff,
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
        (-2, "-2.00 B"),
        (-1023.0, "-1,023.00 B"),
        (-2048, "-2.00 KB"),
        (-2097152, "-2.00 MB"),
        (-2147483648, "-2.00 GB"),
        (-2199023255552, "-2.00 TB"),
        (-2251799813685248, "-2.00 PB"),
        (-2305843009213693952, "-2,048.00 PB"),
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


def test_str_col_diff_added() -> None:
    assert str_col_diff(100, 110) == "10/100 (10.0000 %) columns have been added"


##################################
#     Tests for str_row_diff     #
##################################


def test_str_row_diff_zero() -> None:
    assert str_row_diff(0, 0) == "0/0 (nan %) row has been removed"


def test_str_row_diff_one() -> None:
    assert str_row_diff(100, 99) == "1/100 (1.0000 %) row has been removed"


def test_str_row_diff_multiple() -> None:
    assert str_row_diff(100, 10) == "90/100 (90.0000 %) rows have been removed"


def test_str_row_diff_added() -> None:
    assert str_row_diff(100, 110) == "10/100 (10.0000 %) rows have been added"


####################################
#     Tests for str_shape_diff     #
####################################


def test_str_shape_diff_zero() -> None:
    assert str_shape_diff(orig=(0, 0), final=(0, 0)) == "DataFrame shape: (0, 0) -> (0, 0)"


def test_str_shape_diff_same_shape() -> None:
    assert str_shape_diff(orig=(100, 5), final=(100, 5)) == "DataFrame shape: (100, 5) -> (100, 5)"


def test_str_shape_diff_cols() -> None:
    assert (
        str_shape_diff(orig=(100, 5), final=(100, 3))
        == "DataFrame shape: (100, 5) -> (100, 3) | 2/5 (40.0000 %) columns have been removed"
    )


def test_str_shape_diff_rows() -> None:
    assert (
        str_shape_diff(orig=(100, 5), final=(80, 5))
        == "DataFrame shape: (100, 5) -> (80, 5) | 20/100 (20.0000 %) rows have been removed"
    )


def test_str_shape_diff_cols_and_rows() -> None:
    assert (
        str_shape_diff(orig=(100, 10), final=(80, 8))
        == "DataFrame shape: (100, 10) -> (80, 8) | 20/100 (20.0000 %) rows have been removed | "
        "2/10 (20.0000 %) columns have been removed"
    )


###################################
#     Tests for str_size_diff     #
###################################


def test_str_size_diff_zero() -> None:
    assert (
        str_size_diff(orig=0, final=0)
        == "DataFrame estimated size: 0.00 B -> 0.00 B | difference: 0.00 B (nan %)"
    )


def test_str_size_diff_increase() -> None:
    assert (
        str_size_diff(orig=100, final=120)
        == "DataFrame estimated size: 100.00 B -> 120.00 B | difference: 20.00 B (20.0000 %)"
    )


def test_str_size_diff_decrease() -> None:
    assert (
        str_size_diff(orig=100, final=80)
        == "DataFrame estimated size: 100.00 B -> 80.00 B | difference: -20.00 B (-20.0000 %)"
    )


########################################
#     Tests for str_dataframe_diff     #
########################################


def test_str_dataframe_diff_same() -> None:
    frame1 = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64, "col4": pl.String},
    )
    frame2 = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64, "col4": pl.String},
    )
    assert (
        str_dataframe_diff(orig=frame1, final=frame2) == "DataFrame shape and size did not changed"
    )


def test_str_dataframe_diff_type() -> None:
    frame1 = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64, "col4": pl.String},
    )
    frame2 = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float32, "col4": pl.String},
    )
    assert (
        str_dataframe_diff(orig=frame1, final=frame2)
        == "DataFrame estimated size: 90.00 B -> 70.00 B | difference: -20.00 B (-22.2222 %)"
    )


def test_str_dataframe_diff_shape() -> None:
    frame1 = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64, "col4": pl.String},
    )
    frame2 = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, 6],
            "col2": ["1", "2", "3", "4", "5", "6"],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "col4": ["a", "b", "c", "d", "e", "f"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64, "col4": pl.String},
    )
    assert str_dataframe_diff(orig=frame1, final=frame2).startswith(
        "DataFrame shape: (5, 4) -> (6, 4)"
    )


##############################################
#     Tests for str_boolean_series_stats     #
##############################################


def test_str_boolean_series_stats() -> None:
    assert (
        str_boolean_series_stats(pl.Series([True, False, None, None, False, None]))
        == "true: 1/3 (33.3333 %) | null: 3/6 (50.0000 %)"
    )


def test_str_boolean_series_stats_empty() -> None:
    assert (
        str_boolean_series_stats(pl.Series(dtype=pl.Boolean))
        == "true: 0/0 (nan %) | null: 0/0 (nan %)"
    )


def test_str_boolean_series_stats_all_nulls() -> None:
    assert (
        str_boolean_series_stats(pl.Series([None, None, None, None], dtype=pl.Boolean))
        == "true: 0/0 (nan %) | null: 4/4 (100.0000 %)"
    )


def test_str_boolean_series_stats_all_false() -> None:
    assert (
        str_boolean_series_stats(pl.Series([False, False, False, False]))
        == "true: 0/4 (0.0000 %) | null: 0/4 (0.0000 %)"
    )


def test_str_boolean_series_stats_all_true() -> None:
    assert (
        str_boolean_series_stats(pl.Series([True, True, True, True]))
        == "true: 4/4 (100.0000 %) | null: 0/4 (0.0000 %)"
    )


def test_str_boolean_series_stats_incorrect_dtype() -> None:
    with pytest.raises(ValueError, match=r"Incorrect dtype"):
        str_boolean_series_stats(pl.Series())
