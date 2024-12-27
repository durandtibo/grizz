from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.transformer import MinHorizontal


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [9, 5, 4, 9, 6],
            "col2": [8, 0, 1, 8, 9],
            "col3": [0, 4, 8, 7, 0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64, "col4": pl.String},
    )


##############################################
#     Tests for MinHorizontalTransformer     #
##############################################


def test_min_horizontal_transformer_repr() -> None:
    assert (
        repr(MinHorizontal(columns=["col1", "col3"], out_col="out"))
        == "MinHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_min_horizontal_transformer_str() -> None:
    assert (
        str(MinHorizontal(columns=["col1", "col3"], out_col="out"))
        == "MinHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_min_horizontal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = MinHorizontal(columns=["col1", "col3"], out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'MinHorizontalTransformer.fit' as there are no parameters available to fit"
    )


def test_min_horizontal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_min_horizontal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MinHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_min_horizontal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_min_horizontal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [0, 4, 4, 7, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_min_horizontal_transformer_transform_1_col(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(columns=["col1"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [9, 5, 4, 9, 6],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_min_horizontal_transformer_transform_2_cols(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [0, 4, 4, 7, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_min_horizontal_transformer_transform_3_cols(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [0, 0, 1, 7, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_min_horizontal_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(columns=None, exclude_columns=["col4", "col5"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [0, 0, 1, 7, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_min_horizontal_transformer_transform_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [0, 4, 4, 7, 0],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


def test_min_horizontal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MinHorizontal(columns=["col1", "col3"], out_col="col2")
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_min_horizontal_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [0, 4, 4, 7, 0],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


def test_min_horizontal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MinHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [0, 4, 4, 7, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_min_horizontal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MinHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_min_horizontal_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MinHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [0, 4, 4, 7, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )
