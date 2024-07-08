from __future__ import annotations

import polars as pl
import pytest

from grizz.utils.column import find_common_columns, find_missing_columns


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
        }
    )


#########################################
#     Tests for find_common_columns     #
#########################################


def test_find_common_columns_dataframe(dataframe: pl.DataFrame) -> None:
    assert find_common_columns(dataframe, columns=["col1", "col2", "col3", "col4"]) == (
        "col1",
        "col2",
        "col3",
    )


def test_find_common_columns_1() -> None:
    assert find_common_columns(["col1", "col2", "col3"], columns=["col1"]) == ("col1",)


def test_find_common_columns_2() -> None:
    assert find_common_columns(["col1", "col2", "col3"], columns=["col1", "col2"]) == (
        "col1",
        "col2",
    )


def test_find_common_columns_3() -> None:
    assert find_common_columns(["col1", "col2", "col3"], columns=["col1", "col2", "col3"]) == (
        "col1",
        "col2",
        "col3",
    )


def test_find_common_columns_4() -> None:
    assert find_common_columns(
        ["col1", "col2", "col3"], columns=["col1", "col2", "col3", "col4"]
    ) == ("col1", "col2", "col3")


def test_find_common_columns_empty() -> None:
    assert find_common_columns([], []) == ()


##########################################
#     Tests for find_missing_columns     #
##########################################


def test_find_missing_columns_dataframe(dataframe: pl.DataFrame) -> None:
    assert find_missing_columns(dataframe, columns=["col1", "col2", "col3", "col4"]) == ("col4",)


def test_find_missing_columns_1() -> None:
    assert find_missing_columns(["col1", "col2", "col3"], columns=["col1"]) == ()


def test_find_missing_columns_2() -> None:
    assert find_missing_columns(["col1", "col2", "col3"], columns=["col1", "col2"]) == ()


def test_find_missing_columns_3() -> None:
    assert find_missing_columns(["col1", "col2", "col3"], columns=["col1", "col2", "col3"]) == ()


def test_find_missing_columns_4() -> None:
    assert find_missing_columns(
        ["col1", "col2", "col3"], columns=["col1", "col2", "col3", "col4"]
    ) == ("col4",)


def test_find_missing_columns_empty() -> None:
    assert find_missing_columns([], []) == ()
