from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.lazy.transformer import ConcatColumns


@pytest.fixture
def dataframe() -> pl.LazyFrame:
    return pl.LazyFrame(
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


def test_concat_columns_transformer_repr() -> None:
    assert (
        repr(ConcatColumns(columns=["col1", "col3"], out_col="out"))
        == "ConcatColumnsTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_concat_columns_transformer_str() -> None:
    assert (
        str(ConcatColumns(columns=["col1", "col3"], out_col="out"))
        == "ConcatColumnsTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_concat_columns_transformer_equal_true() -> None:
    assert ConcatColumns(columns=["col1", "col3"], out_col="out").equal(
        ConcatColumns(columns=["col1", "col3"], out_col="out")
    )


def test_concat_columns_transformer_equal_false_different_columns() -> None:
    assert not ConcatColumns(columns=["col1", "col3"], out_col="out").equal(
        ConcatColumns(columns=["col1", "col2", "col3"], out_col="out")
    )


def test_concat_columns_transformer_equal_false_different_out_col() -> None:
    assert not ConcatColumns(columns=["col1", "col3"], out_col="out").equal(
        ConcatColumns(columns=["col1", "col3"], out_col="col")
    )


def test_concat_columns_transformer_equal_false_different_exclude_columns() -> None:
    assert not ConcatColumns(columns=["col1", "col3"], out_col="out").equal(
        ConcatColumns(columns=["col1", "col3"], out_col="out", exclude_columns=["col2"])
    )


def test_concat_columns_transformer_equal_false_different_exist_policy() -> None:
    assert not ConcatColumns(columns=["col1", "col3"], out_col="out").equal(
        ConcatColumns(columns=["col1", "col3"], out_col="out", exist_policy="warn")
    )


def test_concat_columns_transformer_equal_false_different_missing_policy() -> None:
    assert not ConcatColumns(columns=["col1", "col3"], out_col="out").equal(
        ConcatColumns(columns=["col1", "col3"], out_col="out", missing_policy="warn")
    )


def test_concat_columns_transformer_equal_false_different_type() -> None:
    assert not ConcatColumns(columns=["col1", "col3"], out_col="out").equal(42)


def test_concat_columns_transformer_get_args() -> None:
    assert objects_are_equal(
        ConcatColumns(columns=["col1", "col3"], out_col="out").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "out_col": "out",
        },
    )


def test_concat_columns_transformer_fit(
    dataframe: pl.LazyFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ConcatColumnsTransformer.fit' as there are no parameters available to fit"
    )


def test_concat_columns_transformer_fit_missing_policy_ignore(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_concat_columns_transformer_fit_missing_policy_raise(
    dataframe: pl.LazyFrame,
) -> None:
    transformer = ConcatColumns(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_concat_columns_transformer_fit_missing_policy_warn(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_concat_columns_transformer_fit_transform(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
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


def test_concat_columns_transformer_transform_1_col(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(columns=["col1"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
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


def test_concat_columns_transformer_transform_2_cols(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
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


def test_concat_columns_transformer_transform_3_cols(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col2", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
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


def test_concat_columns_transformer_transform_exclude_columns(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(columns=None, exclude_columns=["col4", "col5"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
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


def test_concat_columns_transformer_transform_exist_policy_ignore(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_col="col2", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [[11, 31], [12, 32], [13, 33], [14, 34], [15, 35]],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.List(pl.Int64),
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


def test_concat_columns_transformer_transform_exist_policy_raise(
    dataframe: pl.LazyFrame,
) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_col="col2")
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_concat_columns_transformer_transform_exist_policy_warn(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(columns=["col1", "col3"], out_col="col2", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [[11, 31], [12, 32], [13, 33], [14, 34], [15, 35]],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.List(pl.Int64),
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


def test_concat_columns_transformer_transform_missing_policy_ignore(
    dataframe: pl.LazyFrame,
) -> None:
    transformer = ConcatColumns(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
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


def test_concat_columns_transformer_transform_missing_policy_raise(
    dataframe: pl.LazyFrame,
) -> None:
    transformer = ConcatColumns(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_concat_columns_transformer_transform_missing_policy_warn(dataframe: pl.LazyFrame) -> None:
    transformer = ConcatColumns(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
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
