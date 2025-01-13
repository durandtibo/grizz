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
from grizz.transformer import MaxHorizontal


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
#     Tests for MaxHorizontalTransformer     #
##############################################


def test_max_horizontal_transformer_repr() -> None:
    assert (
        repr(MaxHorizontal(columns=["col1", "col3"], out_col="out"))
        == "MaxHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_max_horizontal_transformer_str() -> None:
    assert (
        str(MaxHorizontal(columns=["col1", "col3"], out_col="out"))
        == "MaxHorizontalTransformer(columns=('col1', 'col3'), out_col='out', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_max_horizontal_transformer_equal_true() -> None:
    assert MaxHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MaxHorizontal(columns=["col1", "col3"], out_col="out")
    )


def test_max_horizontal_transformer_equal_false_different_columns() -> None:
    assert not MaxHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MaxHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    )


def test_max_horizontal_transformer_equal_false_different_out_col() -> None:
    assert not MaxHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MaxHorizontal(columns=["col1", "col3"], out_col="col")
    )


def test_max_horizontal_transformer_equal_false_different_exclude_columns() -> None:
    assert not MaxHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MaxHorizontal(columns=["col1", "col3"], out_col="out", exclude_columns=["col2"])
    )


def test_max_horizontal_transformer_equal_false_different_exist_policy() -> None:
    assert not MaxHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MaxHorizontal(columns=["col1", "col3"], out_col="out", exist_policy="warn")
    )


def test_max_horizontal_transformer_equal_false_different_missing_policy() -> None:
    assert not MaxHorizontal(columns=["col1", "col3"], out_col="out").equal(
        MaxHorizontal(columns=["col1", "col3"], out_col="out", missing_policy="warn")
    )


def test_max_horizontal_transformer_equal_false_different_type() -> None:
    assert not MaxHorizontal(columns=["col1", "col3"], out_col="out").equal(42)


def test_max_horizontal_transformer_get_args() -> None:
    assert objects_are_equal(
        MaxHorizontal(columns=["col1", "col3"], out_col="out").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "out_col": "out",
        },
    )


def test_max_horizontal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3"], out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'MaxHorizontalTransformer.fit' as there are no parameters available to fit"
    )


def test_max_horizontal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_max_horizontal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_max_horizontal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(
        columns=["col1", "col3", "col5"], out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_max_horizontal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [9, 5, 8, 9, 6],
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


def test_max_horizontal_transformer_transform_1_col(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(columns=["col1"], out_col="out")
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


def test_max_horizontal_transformer_transform_2_cols(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [9, 5, 8, 9, 6],
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


def test_max_horizontal_transformer_transform_3_cols(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(columns=["col1", "col2", "col3"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [9, 5, 8, 9, 9],
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


def test_max_horizontal_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(columns=None, exclude_columns=["col4", "col5"], out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [8, 0, 1, 8, 9],
                "col3": [0, 4, 8, 7, 0],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [9, 5, 8, 9, 9],
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


def test_max_horizontal_transformer_transform_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [9, 5, 4, 9, 6],
                "col2": [9, 5, 8, 9, 6],
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


def test_max_horizontal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3"], out_col="col2")
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_max_horizontal_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3"], out_col="col2", exist_policy="warn")
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
                "col2": [9, 5, 8, 9, 6],
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


def test_max_horizontal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxHorizontal(
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
                "out": [9, 5, 8, 9, 6],
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


def test_max_horizontal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxHorizontal(columns=["col1", "col3", "col5"], out_col="out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_max_horizontal_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = MaxHorizontal(
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
                "out": [9, 5, 8, 9, 6],
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
