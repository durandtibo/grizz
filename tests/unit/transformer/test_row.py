from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import FirstRow


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


#########################################
#     Tests for FirstRowTransformer     #
#########################################


def test_first_row_transformer_repr() -> None:
    assert repr(FirstRow(n=3)) == "FirstRowTransformer(n=3)"


def test_first_row_transformer_str() -> None:
    assert str(FirstRow(n=3)) == "FirstRowTransformer(n=3)"


def test_first_row_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = FirstRow(n=3)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'FirstRowTransformer.fit' as there are no parameters available to fit"
    )


def test_first_row_fit_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = FirstRow(n=3)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["1", "2", "3"],
                "col3": ["1", "2", "3"],
                "col4": ["a", "b", "c"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_first_row_transformer_transform_n_2(dataframe: pl.DataFrame) -> None:
    transformer = FirstRow(n=2)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2],
                "col2": ["1", "2"],
                "col3": ["1", "2"],
                "col4": ["a", "b"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_first_row_transformer_transform_n_3(dataframe: pl.DataFrame) -> None:
    transformer = FirstRow(n=3)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["1", "2", "3"],
                "col3": ["1", "2", "3"],
                "col4": ["a", "b", "c"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_first_row_transformer_transform_empty_row() -> None:
    transformer = FirstRow(n=3)
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))
