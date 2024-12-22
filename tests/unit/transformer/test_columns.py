from __future__ import annotations

import polars as pl
import pytest

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer.columns import check_missing_columns


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


###########################################
#     Tests for check_missing_columns     #
###########################################


@pytest.mark.parametrize("missing_ok", [True, False])
def test_check_missing_columns(dataframe: pl.DataFrame, missing_ok: bool) -> None:
    check_missing_columns(dataframe, columns=["col1", "col4"], missing_ok=missing_ok)


def test_check_missing_columns_missing_ok_true(dataframe: pl.DataFrame) -> None:
    with pytest.warns(
        ColumnNotFoundWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        check_missing_columns(dataframe, columns=["col1", "col5"], missing_ok=True)


def test_check_missing_columns_missing_ok_false(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ColumnNotFoundError, match="1 columns are missing in the DataFrame:"):
        check_missing_columns(dataframe, columns=["col1", "col5"])
