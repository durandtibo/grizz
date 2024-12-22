from __future__ import annotations

import polars as pl
import pytest

from grizz.transformer.columns import check_existing_columns, check_missing_columns


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )


############################################
#     Tests for check_existing_columns     #
############################################


@pytest.mark.parametrize("exist_ok", [True, False])
def test_check_existing_columns(dataframe: pl.DataFrame, exist_ok: bool) -> None:
    check_existing_columns(dataframe, columns=["col10"], exist_ok=exist_ok)


def test_check_existing_columns_exist_ok_true(dataframe: pl.DataFrame) -> None:
    with pytest.warns(
        RuntimeWarning, match="1 columns already exist in the DataFrame and will be overwritten:"
    ):
        check_existing_columns(dataframe, columns=["col1", "col5"], exist_ok=True)


def test_check_existing_columns_exist_ok_false(dataframe: pl.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="1 columns already exist in the DataFrame:"):
        check_existing_columns(dataframe, columns=["col1", "col5"])


###########################################
#     Tests for check_missing_columns     #
###########################################


@pytest.mark.parametrize("ignore_missing", [True, False])
def test_check_missing_columns_no_missing(ignore_missing: bool) -> None:
    check_missing_columns(missing_cols=[], ignore_missing=ignore_missing)


def test_check_missing_columns_missing_ignore_missing_true() -> None:
    with pytest.warns(
        RuntimeWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        check_missing_columns(missing_cols=["col5"], ignore_missing=True)


def test_check_missing_columns_missing_ignore_missing_false() -> None:
    with pytest.raises(RuntimeError, match="1 columns are missing in the DataFrame:"):
        check_missing_columns(missing_cols=["col5"])
