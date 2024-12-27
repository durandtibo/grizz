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
from grizz.transformer import SumHorizontal


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
#     Tests for SumHorizontalTransformer     #
##############################################


def test_sum_horizontal_transformer_repr() -> None:
    assert (
        repr(SumHorizontal(columns=["col1", "col3"], out_col="out"))
        == "SumHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), ignore_nulls=True, exist_policy='raise', missing_policy='raise')"
    )


def test_sum_horizontal_transformer_str() -> None:
    assert (
        str(SumHorizontal(columns=["col1", "col3"], out_col="out"))
        == "SumHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), ignore_nulls=True, exist_policy='raise', missing_policy='raise')"
    )


def test_sum_horizontal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = SumHorizontal(columns=["col1", "col3"], out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'SumHorizontalTransformer.fit' as there are no parameters available to fit"
    )


def test_sum_horizontal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_sum_horizontal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SumHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_sum_horizontal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_sum_horizontal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [42, 44, 46, 48, 50],
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


def test_sum_horizontal_transformer_transform_1_col(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=["col1"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [11, 12, 13, 14, 15],
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


def test_sum_horizontal_transformer_transform_2_cols(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [42, 44, 46, 48, 50],
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


def test_sum_horizontal_transformer_transform_3_cols(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [63, 66, 69, 72, 75],
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


def test_sum_horizontal_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=None, exclude_columns=["col4", "col5"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [63, 66, 69, 72, 75],
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


def test_sum_horizontal_transformer_transform_ignore_nulls_false() -> None:
    frame = pl.DataFrame(
        {
            "col1": [None, 12, 13, 14, None],
            "col2": [21, None, 23, None, 25],
            "col3": [31, 32, None, 34, 35],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={
            "col1": pl.Int64,
            "col2": pl.Int64,
            "col3": pl.Int64,
            "col4": pl.String,
        },
    )
    transformer = SumHorizontal(columns=["col1", "col2", "col3"], ignore_nulls=False, out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 12, 13, 14, None],
                "col2": [21, None, 23, None, 25],
                "col3": [31, 32, None, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [None, None, None, None, None],
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


def test_sum_horizontal_transformer_transform_ignore_nulls_true() -> None:
    frame = pl.DataFrame(
        {
            "col1": [None, 12, 13, 14, None],
            "col2": [21, None, 23, None, 25],
            "col3": [31, 32, None, 34, 35],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={
            "col1": pl.Int64,
            "col2": pl.Int64,
            "col3": pl.Int64,
            "col4": pl.String,
        },
    )
    transformer = SumHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 12, 13, 14, None],
                "col2": [21, None, 23, None, 25],
                "col3": [31, 32, None, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [52, 44, 36, 48, 60],
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


def test_sum_horizontal_transformer_transform_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [42, 44, 46, 48, 50],
                "col3": [31, 32, 33, 34, 35],
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


def test_sum_horizontal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SumHorizontal(columns=["col1", "col3"], out_col="col2")
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_sum_horizontal_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [42, 44, 46, 48, 50],
                "col3": [31, 32, 33, 34, 35],
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


def test_sum_horizontal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SumHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [42, 44, 46, 48, 50],
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


def test_sum_horizontal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SumHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_sum_horizontal_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(
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
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [42, 44, 46, 48, 50],
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


def test_sum_horizontal_transformer_transform_missing_all_columns(dataframe: pl.DataFrame) -> None:
    transformer = SumHorizontal(columns=["col5", "col6"], out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="2 columns are missing in the DataFrame and will be ignored:"
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
                "out": [None, None, None, None, None],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Null,
            },
        ),
    )
