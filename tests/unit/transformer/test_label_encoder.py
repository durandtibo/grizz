from __future__ import annotations

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
from grizz.testing.fixture import sklearn_available
from grizz.transformer import LabelEncoder
from grizz.utils.imports import is_sklearn_available

if is_sklearn_available():
    import sklearn


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": ["paris", "paris", "tokyo", "amsterdam", "tokyo"],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.String, "col2": pl.Int64, "col3": pl.String, "col4": pl.String},
    )


#############################################
#     Tests for LabelEncoderTransformer     #
#############################################


@sklearn_available
def test_label_encoder_transformer_repr() -> None:
    assert repr(LabelEncoder(in_col="col1", out_col="out")) == (
        "LabelEncoderTransformer(in_col='col1', out_col='out', exist_policy='raise', "
        "missing_policy='raise')"
    )


@sklearn_available
def test_label_encoder_transformer_str() -> None:
    assert str(LabelEncoder(in_col="col1", out_col="out")) == (
        "LabelEncoderTransformer(in_col='col1', out_col='out', exist_policy='raise', "
        "missing_policy='raise')"
    )


@sklearn_available
def test_label_encoder_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = LabelEncoder(in_col="col1", out_col="out")
    transformer.fit(dataframe)
    assert objects_are_equal(list(transformer._encoder.classes_), ["amsterdam", "paris", "tokyo"])


@sklearn_available
def test_label_encoder_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="in", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)
    assert not hasattr(transformer._encoder, "classes_")


@sklearn_available
def test_label_encoder_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="in", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.fit(dataframe)


@sklearn_available
def test_label_encoder_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="in", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)
    assert not hasattr(transformer._encoder, "classes_")


@sklearn_available
def test_label_encoder_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = LabelEncoder(in_col="col1", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert objects_are_equal(list(transformer._encoder.classes_), ["amsterdam", "paris", "tokyo"])
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["paris", "paris", "tokyo", "amsterdam", "tokyo"],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [1, 1, 2, 0, 2],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Int64,
                "col3": pl.String,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


@sklearn_available
def test_label_encoder_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = LabelEncoder(in_col="col1", out_col="out")
    transformer.fit(dataframe)
    assert objects_are_equal(list(transformer._encoder.classes_), ["amsterdam", "paris", "tokyo"])

    frame = pl.DataFrame(
        {"col1": ["tokyo", "amsterdam", "paris", "tokyo"]}, schema={"col1": pl.String}
    )
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["tokyo", "amsterdam", "paris", "tokyo"],
                "out": [2, 0, 1, 2],
            },
            schema={"col1": pl.String, "out": pl.Int64},
        ),
    )


@sklearn_available
def test_label_encoder_transformer_transform_not_fitted(dataframe: pl.DataFrame) -> None:
    transformer = LabelEncoder(in_col="col1", out_col="out")
    with pytest.raises(
        sklearn.exceptions.NotFittedError, match="This LabelEncoder instance is not fitted yet."
    ):
        transformer.transform(dataframe)


@sklearn_available
def test_label_encoder_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="in", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["paris", "paris", "tokyo", "amsterdam", "tokyo"],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Int64,
                "col3": pl.String,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_label_encoder_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="in", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.transform(dataframe)


@sklearn_available
def test_label_encoder_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="in", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["paris", "paris", "tokyo", "amsterdam", "tokyo"],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Int64,
                "col3": pl.String,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_label_encoder_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="col1", out_col="col2", exist_policy="ignore")
    transformer.fit(pl.DataFrame({"col1": ["tokyo", "amsterdam", "paris", "tokyo"]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["paris", "paris", "tokyo", "amsterdam", "tokyo"],
                "col2": [1, 1, 2, 0, 2],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Int64,
                "col3": pl.String,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_label_encoder_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="col1", out_col="col2")
    transformer.fit(pl.DataFrame({"col1": ["tokyo", "amsterdam", "paris", "tokyo"]}))
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


@sklearn_available
def test_label_encoder_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LabelEncoder(in_col="col1", out_col="col2", exist_policy="warn")
    transformer.fit(pl.DataFrame({"col1": ["tokyo", "amsterdam", "paris", "tokyo"]}))
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["paris", "paris", "tokyo", "amsterdam", "tokyo"],
                "col2": [1, 1, 2, 0, 2],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Int64,
                "col3": pl.String,
                "col4": pl.String,
            },
        ),
    )
