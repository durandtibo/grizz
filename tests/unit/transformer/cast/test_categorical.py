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
from grizz.transformer import CategoricalCast, InplaceCategoricalCast


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


################################################
#     Tests for CategoricalCastTransformer     #
################################################


def test_categorical_cast_transformer_repr() -> None:
    assert repr(CategoricalCast(in_col="col4", out_col="out")) == (
        "CategoricalCastTransformer(in_col='col4', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_categorical_cast_transformer_repr_with_kwargs() -> None:
    assert repr(CategoricalCast(in_col="col4", out_col="out", ordering="lexical")) == (
        "CategoricalCastTransformer(in_col='col4', out_col='out', "
        "exist_policy='raise', missing_policy='raise', ordering='lexical')"
    )


def test_categorical_cast_transformer_str() -> None:
    assert str(CategoricalCast(in_col="col4", out_col="out")) == (
        "CategoricalCastTransformer(in_col='col4', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_categorical_cast_transformer_str_with_kwargs() -> None:
    assert str(CategoricalCast(in_col="col4", out_col="out", ordering="lexical")) == (
        "CategoricalCastTransformer(in_col='col4', out_col='out', "
        "exist_policy='raise', missing_policy='raise', ordering='lexical')"
    )


def test_categorical_cast_transformer_equal_true() -> None:
    assert CategoricalCast(in_col="col4", out_col="out").equal(
        CategoricalCast(in_col="col4", out_col="out")
    )


def test_categorical_cast_transformer_equal_false_different_in_col() -> None:
    assert not CategoricalCast(in_col="col4", out_col="out").equal(
        CategoricalCast(in_col="col3", out_col="out")
    )


def test_categorical_cast_transformer_equal_false_different_out_col() -> None:
    assert not CategoricalCast(in_col="col4", out_col="out").equal(
        CategoricalCast(in_col="col4", out_col="out2")
    )


def test_categorical_cast_transformer_equal_false_different_exist_policy() -> None:
    assert not CategoricalCast(in_col="col4", out_col="out").equal(
        CategoricalCast(in_col="col4", out_col="out", exist_policy="warn")
    )


def test_categorical_cast_transformer_equal_false_different_missing_policy() -> None:
    assert not CategoricalCast(in_col="col4", out_col="out").equal(
        CategoricalCast(in_col="col4", out_col="out", missing_policy="warn")
    )


def test_categorical_cast_transformer_equal_false_different_kwargs() -> None:
    assert not CategoricalCast(in_col="col4", out_col="out").equal(
        CategoricalCast(in_col="col4", out_col="out", ordering="lexical")
    )


def test_categorical_cast_transformer_equal_false_different_type() -> None:
    assert not CategoricalCast(in_col="col4", out_col="out").equal(42)


def test_categorical_cast_transformer_get_args() -> None:
    assert objects_are_equal(
        CategoricalCast(in_col="col4", out_col="out", ordering="lexical").get_args(),
        {
            "in_col": "col4",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
            "ordering": "lexical",
        },
    )


def test_categorical_cast_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = CategoricalCast(in_col="col4", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'CategoricalCastTransformer.fit' as there are no parameters available to fit"
    )


def test_categorical_cast_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = CategoricalCast(in_col="col", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_categorical_cast_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CategoricalCast(in_col="col", out_col="out")
    with pytest.raises(ColumnNotFoundError, match=r"column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_categorical_cast_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = CategoricalCast(in_col="col", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_categorical_cast_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = CategoricalCast(in_col="col4", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "out": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "out": pl.Categorical,
            },
        ),
    )


def test_categorical_cast_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = CategoricalCast(in_col="col4", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "out": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "out": pl.Categorical,
            },
        ),
    )


def test_categorical_cast_transformer_transform_nulls() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["bear", None, "cat", None],
            "col2": [1.0, 1.0, None, None],
        },
        schema={"col1": pl.String, "col2": pl.Float32},
    )
    transformer = CategoricalCast(in_col="col1", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["bear", None, "cat", None],
                "col2": [1.0, 1.0, None, None],
                "out": ["bear", None, "cat", None],
            },
            schema={"col1": pl.String, "col2": pl.Float32, "out": pl.Categorical},
        ),
    )


def test_categorical_cast_transformer_transform_ordering(dataframe: pl.DataFrame) -> None:
    transformer = CategoricalCast(in_col="col4", out_col="out", ordering="lexical")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "out": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "out": pl.Categorical(ordering="lexical"),
            },
        ),
    )


def test_categorical_cast_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CategoricalCast(in_col="col4", out_col="col3", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Categorical, "col4": pl.String},
        ),
    )


def test_categorical_cast_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CategoricalCast(in_col="col4", out_col="col3")
    with pytest.raises(ColumnExistsError, match=r"column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_categorical_cast_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CategoricalCast(in_col="col4", out_col="col3", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match=r"column 'col3' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Categorical, "col4": pl.String},
        ),
    )


def test_categorical_cast_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CategoricalCast(
        in_col="col",
        out_col="out",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_categorical_cast_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CategoricalCast(in_col="col", out_col="out")
    with pytest.raises(ColumnNotFoundError, match=r"column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_categorical_cast_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CategoricalCast(in_col="col", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'col' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


#######################################################
#     Tests for InplaceCategoricalCastTransformer     #
#######################################################


def test_inplace_categorical_cast_transformer_repr() -> None:
    assert repr(InplaceCategoricalCast(col="col4")) == (
        "InplaceCategoricalCastTransformer(col='col4', missing_policy='raise')"
    )


def test_inplace_categorical_cast_transformer_repr_with_kwargs() -> None:
    assert repr(InplaceCategoricalCast(col="col4", ordering="lexical")) == (
        "InplaceCategoricalCastTransformer(col='col4', missing_policy='raise', ordering='lexical')"
    )


def test_inplace_categorical_cast_transformer_str() -> None:
    assert str(InplaceCategoricalCast(col="col4")) == (
        "InplaceCategoricalCastTransformer(col='col4', missing_policy='raise')"
    )


def test_inplace_categorical_cast_transformer_str_with_kwargs() -> None:
    assert str(InplaceCategoricalCast(col="col4", ordering="lexical")) == (
        "InplaceCategoricalCastTransformer(col='col4', missing_policy='raise', ordering='lexical')"
    )


def test_inplace_categorical_cast_transformer_equal_true() -> None:
    assert InplaceCategoricalCast(col="col4").equal(InplaceCategoricalCast(col="col4"))


def test_inplace_categorical_cast_transformer_equal_false_different_col() -> None:
    assert not InplaceCategoricalCast(col="col4").equal(InplaceCategoricalCast(col="col3"))


def test_inplace_categorical_cast_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceCategoricalCast(col="col4").equal(
        InplaceCategoricalCast(col="col4", missing_policy="warn")
    )


def test_inplace_categorical_cast_transformer_equal_false_different_kwargs() -> None:
    assert not InplaceCategoricalCast(col="col4").equal(
        InplaceCategoricalCast(col="col4", ordering="lexical")
    )


def test_inplace_categorical_cast_transformer_equal_false_different_type() -> None:
    assert not InplaceCategoricalCast(col="col4").equal(42)


def test_inplace_categorical_cast_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceCategoricalCast(col="col4", ordering="lexical").get_args(),
        {
            "col": "col4",
            "missing_policy": "raise",
            "ordering": "lexical",
        },
    )


def test_inplace_categorical_cast_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = InplaceCategoricalCast(col="col4")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'InplaceCategoricalCastTransformer.fit' as there are no parameters available to fit"
    )


def test_inplace_categorical_cast_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceCategoricalCast(col="col", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_inplace_categorical_cast_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceCategoricalCast(col="col")
    with pytest.raises(ColumnNotFoundError, match=r"column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_inplace_categorical_cast_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceCategoricalCast(col="col", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_inplace_categorical_cast_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceCategoricalCast(col="col4")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.Categorical,
            },
        ),
    )


def test_inplace_categorical_cast_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceCategoricalCast(col="col4")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.Categorical,
            },
        ),
    )


def test_inplace_categorical_cast_transformer_transform_nulls() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["bear", None, "cat", None],
            "col2": [1.0, 1.0, None, None],
        },
        schema={"col1": pl.String, "col2": pl.Float32},
    )
    transformer = InplaceCategoricalCast(col="col1")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": ["bear", None, "cat", None], "col2": [1.0, 1.0, None, None]},
            schema={"col1": pl.Categorical, "col2": pl.Float32},
        ),
    )


def test_inplace_categorical_cast_transformer_transform_ordering(dataframe: pl.DataFrame) -> None:
    transformer = InplaceCategoricalCast(col="col4", ordering="lexical")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.Categorical(ordering="lexical"),
            },
        ),
    )


def test_inplace_categorical_cast_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceCategoricalCast(col="col", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_inplace_categorical_cast_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceCategoricalCast(col="col")
    with pytest.raises(ColumnNotFoundError, match=r"column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_inplace_categorical_cast_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceCategoricalCast(col="col", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'col' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )
