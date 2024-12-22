from __future__ import annotations

import warnings

import polars as pl
import pytest

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.utils.column import (
    check_column_exist_policy,
    check_column_missing_policy,
    check_existing_columns,
    check_missing_columns,
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


@pytest.mark.parametrize("policy", ["ignore", "raise", "warn"])
def test_check_column_exist_policy_valid(policy: str) -> None:
    check_column_exist_policy(policy)


def test_check_column_exist_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'exist_policy': incorrect"):
        check_column_exist_policy("incorrect")


#################################################
#     Tests for check_column_missing_policy     #
#################################################


@pytest.mark.parametrize("policy", ["ignore", "raise", "warn"])
def test_check_column_missing_policy_valid(policy: str) -> None:
    check_column_missing_policy(policy)


def test_check_column_missing_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'missing_policy': incorrect"):
        check_column_missing_policy("incorrect")


############################################
#     Tests for check_existing_columns     #
############################################


@pytest.mark.parametrize("exist_policy", ["ignore", "raise", "warn"])
def test_check_existing_columns(dataframe: pl.DataFrame, exist_policy: str) -> None:
    check_existing_columns(dataframe, columns=["col10"], exist_policy=exist_policy)


def test_check_existing_columns_ignore(dataframe: pl.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_existing_columns(dataframe, columns=["col1", "col5"], exist_policy="ignore")


def test_check_existing_columns_raise(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ColumnExistsError, match="1 columns already exist in the DataFrame:"):
        check_existing_columns(dataframe, columns=["col1", "col5"])


def test_check_existing_columns_warn(dataframe: pl.DataFrame) -> None:
    with pytest.warns(
        ColumnExistsWarning,
        match="1 columns already exist in the DataFrame and will be overwritten:",
    ):
        check_existing_columns(dataframe, columns=["col1", "col5"], exist_policy="warn")


def test_check_existing_columns_exist_policy_incorrect(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="Incorrect 'exist_policy': incorrect"):
        check_existing_columns(dataframe, columns=["col1", "col5"], exist_policy="incorrect")


###########################################
#     Tests for check_missing_columns     #
###########################################


@pytest.mark.parametrize("missing_policy", ["ignore", "raise", "warn"])
def test_check_missing_columns(dataframe: pl.DataFrame, missing_policy: str) -> None:
    check_missing_columns(dataframe, columns=["col1", "col3"], missing_policy=missing_policy)


def test_check_missing_columns_ignore(dataframe: pl.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_missing_columns(dataframe, columns=["col1", "col5"], missing_policy="ignore")


def test_check_missing_columns_raise(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ColumnNotFoundError, match="1 columns are missing in the DataFrame:"):
        check_missing_columns(dataframe, columns=["col1", "col5"])


def test_check_missing_columns_warn(dataframe: pl.DataFrame) -> None:
    with pytest.warns(
        ColumnNotFoundWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        check_missing_columns(dataframe, columns=["col1", "col5"], missing_policy="warn")


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
