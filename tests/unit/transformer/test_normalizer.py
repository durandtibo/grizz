from __future__ import annotations

import logging
import warnings
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.testing.fixture import sklearn_available
from grizz.transformer import Normalizer


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


###########################################
#     Tests for NormalizerTransformer     #
###########################################


@sklearn_available
def test_normarizer_transformer_repr() -> None:
    assert repr(Normalizer(columns=["col1", "col3"], prefix="", suffix="_norm")) == (
        "NormalizerTransformer(columns=('col1', 'col3'), prefix='', suffix='_norm', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


@sklearn_available
def test_normarizer_transformer_str() -> None:
    assert str(Normalizer(columns=["col1", "col3"], prefix="", suffix="_norm")) == (
        "NormalizerTransformer(columns=('col1', 'col3'), prefix='', suffix='_norm', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


@sklearn_available
def test_normarizer_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = Normalizer(columns=["col1", "col3"], prefix="", suffix="_norm")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'NormalizerTransformer.fit' as there are no parameters available to fit"
    )


@sklearn_available
def test_normarizer_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_norm", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


@sklearn_available
def test_normarizer_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(columns=["col1", "col3", "col5"], prefix="", suffix="_norm")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


@sklearn_available
def test_normarizer_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_norm", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


@sklearn_available
def test_normarizer_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Normalizer(columns=["col1", "col3"], prefix="", suffix="_norm")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [5, 4, 3, 2, 1],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_norm": [
                    0.19611613513818404,
                    0.4472135954999579,
                    0.7071067811865476,
                    0.8944271909999159,
                    0.9805806756909202,
                ],
                "col3_norm": [
                    0.9805806756909202,
                    0.8944271909999159,
                    0.7071067811865476,
                    0.4472135954999579,
                    0.19611613513818404,
                ],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_norm": pl.Float64,
                "col3_norm": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_normarizer_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = Normalizer(columns=["col1", "col3"], prefix="", suffix="_norm")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [5, 4, 3, 2, 1],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_norm": [
                    0.19611613513818404,
                    0.4472135954999579,
                    0.7071067811865476,
                    0.8944271909999159,
                    0.9805806756909202,
                ],
                "col3_norm": [
                    0.9805806756909202,
                    0.8944271909999159,
                    0.7071067811865476,
                    0.4472135954999579,
                    0.19611613513818404,
                ],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_norm": pl.Float64,
                "col3_norm": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_normarizer_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [0, 1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [
                0.0,
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
            "col3": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Int64},
    )
    transformer = Normalizer(columns=["col1", "col2", "col3"], prefix="", suffix="_norm")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [0, 1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
                "col2": [
                    0.0,
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
                "col3": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                "col1_norm": [
                    0.0,
                    0.09901475429766744,
                    0.09901475429766744,
                    0.09901475429766744,
                    0.09901475429766744,
                    0.09901475429766743,
                    None,
                    0.014284256782850143,
                    None,
                    None,
                    0.009999500037496875,
                    None,
                ],
                "col2_norm": [
                    0.0,
                    -0.09901475429766744,
                    -0.09901475429766744,
                    -0.09901475429766744,
                    -0.09901475429766744,
                    -0.09901475429766743,
                    0.016664352333993333,
                    None,
                    None,
                    0.011110425303554916,
                    None,
                    None,
                ],
                "col3_norm": [
                    0.0,
                    0.9901475429766744,
                    0.9901475429766744,
                    0.9901475429766744,
                    0.9901475429766744,
                    0.9901475429766743,
                    0.9998611400396,
                    0.9998979747995099,
                    1.0,
                    0.9999382773199424,
                    0.9999500037496876,
                    1.0,
                ],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col1_norm": pl.Float64,
                "col2_norm": pl.Float64,
                "col3_norm": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_normarizer_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    0.19611613513818404,
                    0.4472135954999579,
                    0.7071067811865476,
                    0.8944271909999159,
                    0.9805806756909202,
                ],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    0.9805806756909202,
                    0.8944271909999159,
                    0.7071067811865476,
                    0.4472135954999579,
                    0.19611613513818404,
                ],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float64,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_normarizer_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_normarizer_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    0.19611613513818404,
                    0.4472135954999579,
                    0.7071067811865476,
                    0.8944271909999159,
                    0.9805806756909202,
                ],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    0.9805806756909202,
                    0.8944271909999159,
                    0.7071067811865476,
                    0.4472135954999579,
                    0.19611613513818404,
                ],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float64,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_normarizer_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_norm", missing_policy="ignore"
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
                "col1_norm": [
                    0.19611613513818404,
                    0.4472135954999579,
                    0.7071067811865476,
                    0.8944271909999159,
                    0.9805806756909202,
                ],
                "col3_norm": [
                    0.9805806756909202,
                    0.8944271909999159,
                    0.7071067811865476,
                    0.4472135954999579,
                    0.19611613513818404,
                ],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_norm": pl.Float64,
                "col3_norm": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_normarizer_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(columns=["col1", "col3", "col5"], prefix="", suffix="_norm")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_normarizer_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Normalizer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_norm", missing_policy="warn"
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
                "col1_norm": [
                    0.19611613513818404,
                    0.4472135954999579,
                    0.7071067811865476,
                    0.8944271909999159,
                    0.9805806756909202,
                ],
                "col3_norm": [
                    0.9805806756909202,
                    0.8944271909999159,
                    0.7071067811865476,
                    0.4472135954999579,
                    0.19611613513818404,
                ],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_norm": pl.Float64,
                "col3_norm": pl.Float64,
            },
        ),
    )


def test_normarizer_transformer_no_sklearn() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        Normalizer(columns=["col1", "col3"], prefix="", suffix="_norm")
