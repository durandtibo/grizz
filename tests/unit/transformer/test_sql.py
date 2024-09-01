from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import SqlTransformer


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


####################################
#     Tests for SqlTransformer     #
####################################


def test_sql_transformer_repr() -> None:
    assert repr(SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")) == (
        "SqlTransformer(\n  (query): SELECT col1, col4 FROM self WHERE col1 > 2\n)"
    )


def test_sql_transformer_str() -> None:
    assert str(SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")) == (
        "SqlTransformer(\n  (query): SELECT col1, col4 FROM self WHERE col1 > 2\n)"
    )


def test_sql_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [3, 4, 5], "col4": ["c", "d", "e"]},
            schema={"col1": pl.Int64, "col4": pl.String},
        ),
    )
