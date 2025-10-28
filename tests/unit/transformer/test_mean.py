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
from grizz.transformer import MeanHorizontal
from tests.conftest import polars_greater_equal_1_17


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


###############################################
#     Tests for MeanHorizontalTransformer     #
###############################################


def test_mean_horizontal_transformer_repr() -> None:
    assert (
        repr(MeanHorizontal(columns=["col1", "col3"], out_col="out"))
        == "MeanHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_mean_horizontal_transformer_str() -> None:
    assert (
        str(MeanHorizontal(columns=["col1", "col3"], out_col="out"))
        == "MeanHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_mean_horizontal_transformer_equal_true() -> None:
    assert MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MeanHorizontal(columns=["col1", "col3"], out_col="out")
    )


def test_mean_horizontal_transformer_equal_false_different_columns() -> None:
    assert not MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MeanHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    )


def test_mean_horizontal_transformer_equal_false_different_out_col() -> None:
    assert not MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MeanHorizontal(columns=["col1", "col3"], out_col="col")
    )


def test_mean_horizontal_transformer_equal_false_different_exclude_columns() -> None:
    assert not MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MeanHorizontal(columns=["col1", "col3"], out_col="out", exclude_columns=["col2"])
    )


def test_mean_horizontal_transformer_equal_false_different_exist_policy() -> None:
    assert not MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MeanHorizontal(columns=["col1", "col3"], out_col="out", exist_policy="warn")
    )


def test_mean_horizontal_transformer_equal_false_different_missing_policy() -> None:
    assert not MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MeanHorizontal(columns=["col1", "col3"], out_col="out", missing_policy="warn")
    )


def test_mean_horizontal_transformer_equal_false_different_kwargs() -> None:
    assert not MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MeanHorizontal(columns=["col1", "col3"], out_col="out", ignore_nulls=False)
    )


def test_mean_horizontal_transformer_equal_false_different_type() -> None:
    assert not MeanHorizontal(columns=["col1", "col3"], out_col="out").equal(42)


def test_mean_horizontal_transformer_get_args() -> None:
    assert objects_are_equal(
        MeanHorizontal(columns=["col1", "col3"], out_col="out").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "out_col": "out",
        },
    )


def test_mean_horizontal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3"], out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'MeanHorizontalTransformer.fit' as there are no parameters available to fit"
    )


def test_mean_horizontal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_mean_horizontal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_mean_horizontal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_mean_horizontal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [21.0, 22.0, 23.0, 24.0, 25.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_1_col(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=["col1"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [11.0, 12.0, 13.0, 14.0, 15.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_2_cols(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [21.0, 22.0, 23.0, 24.0, 25.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_3_cols(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [21.0, 22.0, 23.0, 24.0, 25.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=None, exclude_columns=["col4", "col5"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21, 22, 23, 24, 25],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [21.0, 22.0, 23.0, 24.0, 25.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


@polars_greater_equal_1_17
def test_mean_horizontal_transformer_transform_ignore_nulls_false() -> None:
    # ignore_nulls argument was added in polars 1.17
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
    transformer = MeanHorizontal(
        columns=["col1", "col2", "col3"], ignore_nulls=False, out_col="out"
    )
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
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_ignore_nulls_true() -> None:
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
    transformer = MeanHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 12, 13, 14, None],
                "col2": [21, None, 23, None, 25],
                "col3": [31, 32, None, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [26.0, 22.0, 18.0, 24.0, 30.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21.0, 22.0, 23.0, 24.0, 25.0],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3"], out_col="col2")
    with pytest.raises(ColumnExistsError, match=r"column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_mean_horizontal_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match=r"column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [11, 12, 13, 14, 15],
                "col2": [21.0, 22.0, 23.0, 24.0, 25.0],
                "col3": [31, 32, 33, 34, 35],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MeanHorizontal(
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
                "out": [21.0, 22.0, 23.0, 24.0, 25.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MeanHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_mean_horizontal_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
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
                "out": [21.0, 22.0, 23.0, 24.0, 25.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int64,
                "col3": pl.Int64,
                "col4": pl.String,
                "out": pl.Float64,
            },
        ),
    )


def test_mean_horizontal_transformer_transform_missing_all_columns(dataframe: pl.DataFrame) -> None:
    transformer = MeanHorizontal(columns=["col5", "col6"], out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"2 columns are missing in the DataFrame and will be ignored:"
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
                "out": pl.Float64,
            },
        ),
    )
