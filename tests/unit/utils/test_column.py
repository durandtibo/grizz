from __future__ import annotations

import warnings

import polars as pl
import pytest

from grizz.exceptions import ColumnExistsError, ColumnExistsWarning
from grizz.utils.column import (
    check_column_exist_policy,
    check_existing_columns,
    find_common_columns,
    find_missing_columns,
)


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
        }
    )


###############################################
#     Tests for check_column_exist_policy     #
###############################################


@pytest.mark.parametrize("nan_policy", ["ignore", "raise", "warn"])
def test_check_column_exist_policy_valid(nan_policy: str) -> None:
    check_column_exist_policy(nan_policy)


def test_check_column_exist_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'col_exist_policy': incorrect"):
        check_column_exist_policy("incorrect")


############################################
#     Tests for check_existing_columns     #
############################################


@pytest.mark.parametrize("col_exist_policy", ["ignore", "raise", "warn"])
def test_check_existing_columns(dataframe: pl.DataFrame, col_exist_policy: str) -> None:
    check_existing_columns(dataframe, columns=["col10"], col_exist_policy=col_exist_policy)


def test_check_existing_columns_col_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_existing_columns(dataframe, columns=["col1", "col5"], col_exist_policy="ignore")


def test_check_existing_columns_col_exist_policy_raise(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ColumnExistsError, match="1 columns already exist in the DataFrame:"):
        check_existing_columns(dataframe, columns=["col1", "col5"])


def test_check_existing_columns_col_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    with pytest.warns(
        ColumnExistsWarning,
        match="1 columns already exist in the DataFrame and will be overwritten:",
    ):
        check_existing_columns(dataframe, columns=["col1", "col5"], col_exist_policy="warn")


def test_check_existing_columns_col_exist_policy_incorrect(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="Incorrect 'col_exist_policy': incorrect"):
        check_existing_columns(dataframe, columns=["col1", "col5"], col_exist_policy="incorrect")


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
