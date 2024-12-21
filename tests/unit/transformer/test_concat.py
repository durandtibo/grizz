from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import ConcatColumns


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [11, 12, 13, 14, 15],
            "col2": [21, 22, 23, 24, 25],
            "col3": [31, 32, 33, 34, 35],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64, "col4": pl.String},
    )


##############################################
#     Tests for ConcatColumnsTransformer     #
##############################################


def test_concat_transformer_repr() -> None:
    assert repr(ConcatColumns(columns=["col1", "col3"], out_column="out")).startswith(
        "ConcatColumnsTransformer("
    )


def test_concat_transformer_str() -> None:
    assert str(ConcatColumns(columns=["col1", "col3"], out_column="out")).startswith(
        "ConcatColumnsTransformer("
    )


def test_concat_transformer_fit(dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_column="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ConcatColumnsTransformer.fit' as there are no parameters available to fit"
    )


def test_concat_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_column="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [[11, 31], [12, 32], [13, 33], [14, 34], [15, 35]],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.List(pl.Int64),
            },
        ),
    )


def test_concat_transformer_transform_1_col(dataframe: pl.DataFrame) -> None:
    transformer = ConcatColumns(columns=["col1"], out_column="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [[11], [12], [13], [14], [15]],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.List(pl.Int64),
            },
        ),
    )


def test_concat_transformer_transform_2_cols(dataframe: pl.DataFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_column="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [[11, 31], [12, 32], [13, 33], [14, 34], [15, 35]],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.List(pl.Int64),
            },
        ),
    )


def test_concat_transformer_transform_3_cols(dataframe: pl.DataFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col2", "col3"], out_column="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [[11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34], [15, 25, 35]],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.List(pl.Int64),
            },
        ),
    )


def test_concat_transformer_transform_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ConcatColumns(columns=["col1", "col3", "col5"], out_column="out")
    with pytest.raises(RuntimeError, match="1 columns are missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_concat_transformer_transform_ignore_missing_true(dataframe: pl.DataFrame) -> None:
    transformer = ConcatColumns(
        columns=["col1", "col3", "col5"], out_column="out", ignore_missing=True
    )
    with pytest.warns(
        RuntimeWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [[11, 31], [12, 32], [13, 33], [14, 34], [15, 35]],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.List(pl.Int64),
            },
        ),
    )
