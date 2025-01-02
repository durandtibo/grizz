from __future__ import annotations

import logging
import warnings
from unittest.mock import patch

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
from grizz.testing.fixture import sklearn_available
from grizz.transformer import Binarizer


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "col3": [5, 4, 3, 2, 1],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Float32, "col3": pl.Int64, "col4": pl.String},
    )


##########################################
#     Tests for BinarizerTransformer     #
##########################################


@sklearn_available
def test_binarizer_transformer_repr() -> None:
    assert repr(Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin")) == (
        "BinarizerTransformer(columns=('col1', 'col3'), prefix='', suffix='_bin', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


@sklearn_available
def test_binarizer_transformer_str() -> None:
    assert str(Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin")) == (
        "BinarizerTransformer(columns=('col1', 'col3'), prefix='', suffix='_bin', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_binarizer_transformer_equal_true() -> None:
    assert Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin")
    )


def test_binarizer_transformer_equal_false_different_columns() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col2", "col3"], prefix="", suffix="_bin")
    )


def test_binarizer_transformer_equal_false_different_prefix() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col3"], prefix="bin_", suffix="_bin")
    )


def test_binarizer_transformer_equal_false_different_suffix() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col3"], prefix="", suffix="")
    )


def test_binarizer_transformer_equal_false_different_exclude_columns() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin", exclude_columns=["col4"])
    )


def test_binarizer_transformer_equal_false_different_exist_policy() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin", exist_policy="warn")
    )


def test_binarizer_transformer_equal_false_different_missing_policy() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin", missing_policy="warn")
    )


def test_binarizer_transformer_equal_false_different_kwargs() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(
        Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin", threshold=1.0)
    )


def test_binarizer_transformer_equal_false_different_type() -> None:
    assert not Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin").equal(42)


def test_binarizer_transformer_get_args() -> None:
    assert objects_are_equal(
        Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin", threshold=1.0).get_args(),
        {
            "columns": ("col1", "col3"),
            "prefix": "",
            "suffix": "_bin",
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "threshold": 1.0,
        }
    )


@sklearn_available
def test_binarizer_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'BinarizerTransformer.fit' as there are no parameters available to fit"
    )


@sklearn_available
def test_binarizer_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_bin", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


@sklearn_available
def test_binarizer_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(columns=["col1", "col3", "col5"], prefix="", suffix="_bin")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


@sklearn_available
def test_binarizer_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_bin", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


@sklearn_available
def test_binarizer_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin", threshold=1.5)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [5, 4, 3, 2, 1],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_bin": [0, 1, 1, 1, 1],
                "col3_bin": [1, 1, 1, 1, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_bin": pl.Int64,
                "col3_bin": pl.Int64,
            },
        ),
    )


@sklearn_available
def test_binarizer_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin", threshold=1.5)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [5, 4, 3, 2, 1],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_bin": [0, 1, 1, 1, 1],
                "col3_bin": [1, 1, 1, 1, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_bin": pl.Int64,
                "col3_bin": pl.Int64,
            },
        ),
    )


@sklearn_available
def test_binarizer_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0, 1, None, None, 1, float("nan"), float("nan")],
            "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Int64},
    )
    transformer = Binarizer(
        columns=["col1", "col2", "col3"], prefix="", suffix="_bin", threshold=1.5
    )
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
                "col2": [
                    -1.0,
                    -2.0,
                    -3.0,
                    -4.0,
                    -5.0,
                    1,
                    None,
                    None,
                    1,
                    float("nan"),
                    float("nan"),
                ],
                "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                "col1_bin": [0, 1, 1, 1, 1, None, 0, None, None, 0, None],
                "col2_bin": [0, 0, 0, 0, 0, 0, None, None, 0, None, None],
                "col3_bin": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col1_bin": pl.Float64,
                "col2_bin": pl.Float64,
                "col3_bin": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_binarizer_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore", threshold=1.5
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 1, 1],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [1, 1, 1, 1, 0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_binarizer_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_binarizer_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn", threshold=1.5
    )
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 1, 1],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [1, 1, 1, 1, 0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_binarizer_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(
        columns=["col1", "col3", "col5"],
        prefix="",
        suffix="_bin",
        missing_policy="ignore",
        threshold=1.5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [5, 4, 3, 2, 1],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_bin": [0, 1, 1, 1, 1],
                "col3_bin": [1, 1, 1, 1, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_bin": pl.Int64,
                "col3_bin": pl.Int64,
            },
        ),
    )


@sklearn_available
def test_binarizer_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(columns=["col1", "col3", "col5"], prefix="", suffix="_bin")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_binarizer_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Binarizer(
        columns=["col1", "col3", "col5"],
        prefix="",
        suffix="_bin",
        missing_policy="warn",
        threshold=1.5,
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [5, 4, 3, 2, 1],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_bin": [0, 1, 1, 1, 1],
                "col3_bin": [1, 1, 1, 1, 0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_bin": pl.Int64,
                "col3_bin": pl.Int64,
            },
        ),
    )


def test_binarizer_transformer_no_sklearn() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        Binarizer(columns=["col1", "col3"], prefix="", suffix="_bin")
