from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import Cast, DecimalCast, FloatCast, IntegerCast


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


#####################################
#     Tests for CastTransformer     #
#####################################


def test_cast_transformer_repr() -> None:
    assert repr(Cast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_cast_transformer_repr_with_kwargs() -> None:
    assert repr(Cast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_cast_transformer_str() -> None:
    assert str(Cast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_cast_transformer_str_with_kwargs() -> None:
    assert str(Cast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_cast_transformer_fit(dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'CastTransformer.fit' as there are no parameters available to fit"
    )


def test_cast_transformer_fit_transform_int32(dataframe: pl.DataFrame) -> None:
    transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.String, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_cast_transformer_fit_transform_float32(dataframe: pl.DataFrame) -> None:
    transformer = Cast(columns=["col1", "col2"], dtype=pl.Float32)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_cast_transformer_transform_int32(dataframe: pl.DataFrame) -> None:
    transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.String, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_cast_transformer_transform_float32(dataframe: pl.DataFrame) -> None:
    transformer = Cast(columns=["col1", "col2"], dtype=pl.Float32)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_cast_transformer_transform_empty() -> None:
    transformer = Cast(columns=["col1", "col2"], dtype=pl.Float32)
    out = transformer.transform(
        pl.DataFrame(
            data={"col1": [], "col2": [], "col3": [], "col4": []},
            schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={"col1": [], "col2": [], "col3": [], "col4": []},
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_cast_transformer_transform_columns_none() -> None:
    transformer = Cast(columns=None, dtype=pl.Float32)
    out = transformer.transform(
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": [101.0, 102.0, 103.0, 104.0, 105.0],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Float32, "col4": pl.Float32},
        ),
    )


def test_cast_transformer_transform_exclude_columns() -> None:
    transformer = Cast(columns=None, dtype=pl.Float32, exclude_columns=["col4", "col5"])
    out = transformer.transform(
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Float32, "col4": pl.String},
        ),
    )


def test_cast_transformer_transform_columns_empty() -> None:
    transformer = Cast(columns=[], dtype=pl.Float32)
    out = transformer.transform(
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_cast_transformer_transform_strict_false(dataframe: pl.DataFrame) -> None:
    transformer = Cast(columns=None, dtype=pl.Float32, strict=False)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": [None, None, None, None, None],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Float32, "col4": pl.Float32},
        ),
    )


def test_cast_transformer_transform_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = Cast(columns=["col1", "col3", "col5"], dtype=pl.Float32, missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.String,
                "col3": pl.Float32,
                "col4": pl.String,
            },
        ),
    )


def test_cast_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Cast(columns=["col1", "col3", "col5"], dtype=pl.Float32)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_cast_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = Cast(columns=["col1", "col3", "col5"], dtype=pl.Float32, missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.String,
                "col3": pl.Float32,
                "col4": pl.String,
            },
        ),
    )


############################################
#     Tests for DecimalCastTransformer     #
############################################


@pytest.fixture
def frame_decimal() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Decimal, "col3": pl.Decimal, "col4": pl.String},
    )


def test_decimal_cast_transformer_repr() -> None:
    assert repr(DecimalCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "DecimalCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_decimal_cast_transformer_repr_with_kwargs() -> None:
    assert repr(DecimalCast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "DecimalCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_decimal_cast_transformer_str() -> None:
    assert str(DecimalCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "DecimalCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_decimal_cast_transformer_str_with_kwargs() -> None:
    assert str(DecimalCast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "DecimalCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_decimal_cast_transformer_fit(
    frame_decimal: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DecimalCast(columns=["col1", "col2"], dtype=pl.Int32)
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_decimal)
    assert caplog.messages[0].startswith(
        "Skipping 'DecimalCastTransformer.fit' as there are no parameters available to fit"
    )


def test_decimal_cast_transformer_fit_transform_int32(frame_decimal: pl.DataFrame) -> None:
    transformer = DecimalCast(columns=["col1", "col2"], dtype=pl.Int32)
    out = transformer.fit_transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Decimal, "col4": pl.String},
        ),
    )


def test_decimal_cast_transformer_transform_int32(frame_decimal: pl.DataFrame) -> None:
    transformer = DecimalCast(columns=["col1", "col2"], dtype=pl.Int32)
    out = transformer.transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Decimal, "col4": pl.String},
        ),
    )


def test_decimal_cast_transformer_transform_float32(frame_decimal: pl.DataFrame) -> None:
    transformer = DecimalCast(columns=["col1", "col2"], dtype=pl.Float32)
    out = transformer.transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Float32, "col3": pl.Decimal, "col4": pl.String},
        ),
    )


def test_decimal_cast_transformer_transform_none(frame_decimal: pl.DataFrame) -> None:
    transformer = DecimalCast(columns=None, dtype=pl.Float32)
    out = transformer.transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Float32, "col3": pl.Float32, "col4": pl.String},
        ),
    )


def test_decimal_cast_transformer_transform_exclude_columns(frame_decimal: pl.DataFrame) -> None:
    transformer = DecimalCast(columns=None, dtype=pl.Float32, exclude_columns=["col2", "col5"])
    out = transformer.transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Decimal, "col3": pl.Float32, "col4": pl.String},
        ),
    )


def test_decimal_cast_transformer_transform_strict_false(frame_decimal: pl.DataFrame) -> None:
    transformer = DecimalCast(columns=None, dtype=pl.Float32, strict=False)
    out = transformer.transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Float32, "col3": pl.Float32, "col4": pl.String},
        ),
    )


def test_decimal_cast_transformer_transform_missing_policy_ignore(
    frame_decimal: pl.DataFrame,
) -> None:
    transformer = DecimalCast(
        columns=["col1", "col2", "col5"], dtype=pl.Float32, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Decimal,
                "col4": pl.String,
            },
        ),
    )


def test_decimal_cast_transformer_transform_missing_policy_raise(
    frame_decimal: pl.DataFrame,
) -> None:
    transformer = DecimalCast(columns=["col1", "col3", "col5"], dtype=pl.Float32)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_decimal)


def test_decimal_cast_transformer_transform_missing_policy_warn(
    frame_decimal: pl.DataFrame,
) -> None:
    transformer = DecimalCast(
        columns=["col1", "col2", "col5"], dtype=pl.Float32, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_decimal)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Decimal,
                "col4": pl.String,
            },
        ),
    )


##########################################
#     Tests for FloatCastTransformer     #
##########################################


@pytest.fixture
def frame_float() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.Float64, "col4": pl.String},
    )


def test_float_cast_transformer_repr() -> None:
    assert repr(FloatCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "FloatCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_float_cast_transformer_repr_with_kwargs() -> None:
    assert repr(FloatCast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "FloatCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_float_cast_transformer_str() -> None:
    assert str(FloatCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "FloatCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_float_cast_transformer_str_with_kwargs() -> None:
    assert str(FloatCast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "FloatCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_float_cast_transformer_fit(
    frame_float: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = FloatCast(columns=["col1", "col2"], dtype=pl.Int32)
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_float)
    assert caplog.messages[0].startswith(
        "Skipping 'FloatCastTransformer.fit' as there are no parameters available to fit"
    )


def test_float_cast_transformer_fit_transform_int32(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(columns=["col1", "col2"], dtype=pl.Int32)
    out = transformer.fit_transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64, "col4": pl.String},
        ),
    )


def test_float_cast_transformer_transform_int32(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(columns=["col1", "col2"], dtype=pl.Int32)
    out = transformer.transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64, "col4": pl.String},
        ),
    )


def test_float_cast_transformer_transform_float32(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(columns=["col1", "col2"], dtype=pl.Float32)
    out = transformer.transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Float32, "col3": pl.Float64, "col4": pl.String},
        ),
    )


def test_float_cast_transformer_transform_none(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(columns=None, dtype=pl.Int32)
    out = transformer.transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_float_cast_transformer_transform_exclude_columns(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(columns=None, dtype=pl.Int32, exclude_columns=["col2", "col5"])
    out = transformer.transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_float_cast_transformer_transform_strict_false(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(columns=None, dtype=pl.Int32, strict=False)
    out = transformer.transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_float_cast_transformer_transform_missing_policy_ignore(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(
        columns=["col1", "col2", "col5"], dtype=pl.Int32, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int32,
                "col3": pl.Float64,
                "col4": pl.String,
            },
        ),
    )


def test_float_cast_transformer_transform_missing_policy_raise(
    frame_float: pl.DataFrame,
) -> None:
    transformer = FloatCast(columns=["col1", "col3", "col5"], dtype=pl.Int32)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_float)


def test_float_cast_transformer_transform_missing_policy_warn(frame_float: pl.DataFrame) -> None:
    transformer = FloatCast(columns=["col1", "col2", "col5"], dtype=pl.Int32, missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_float)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Int32,
                "col3": pl.Float64,
                "col4": pl.String,
            },
        ),
    )


############################################
#     Tests for IntegerCastTransformer     #
############################################


@pytest.fixture
def frame_integer() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col3": [1, 2, 3, 4, 5],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.Int64, "col4": pl.String},
    )


def test_integer_cast_transformer_repr() -> None:
    assert repr(IntegerCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "IntegerCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_integer_cast_transformer_repr_with_kwargs() -> None:
    assert repr(IntegerCast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "IntegerCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_integer_cast_transformer_str() -> None:
    assert str(IntegerCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "IntegerCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_integer_cast_transformer_str_with_kwargs() -> None:
    assert str(IntegerCast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "IntegerCastTransformer(columns=('col1', 'col3'), dtype=Int32, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_integer_cast_transformer_fit(
    frame_integer: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = IntegerCast(columns=["col1", "col2"], dtype=pl.Int32)
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_integer)
    assert caplog.messages[0].startswith(
        "Skipping 'IntegerCastTransformer.fit' as there are no parameters available to fit"
    )


def test_integer_cast_transformer_fit_transform_int32(frame_integer: pl.DataFrame) -> None:
    transformer = IntegerCast(columns=["col1", "col2"], dtype=pl.Int32)
    out = transformer.fit_transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.Float64, "col3": pl.Int64, "col4": pl.String},
        ),
    )


def test_integer_cast_transformer_transform_int32(frame_integer: pl.DataFrame) -> None:
    transformer = IntegerCast(columns=["col1", "col2"], dtype=pl.Int32)
    out = transformer.transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.Float64, "col3": pl.Int64, "col4": pl.String},
        ),
    )


def test_integer_cast_transformer_transform_float32(frame_integer: pl.DataFrame) -> None:
    transformer = IntegerCast(columns=["col1", "col2"], dtype=pl.Float32)
    out = transformer.transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float64, "col3": pl.Int64, "col4": pl.String},
        ),
    )


def test_integer_cast_transformer_transform_none(frame_integer: pl.DataFrame) -> None:
    transformer = IntegerCast(columns=None, dtype=pl.Int32)
    out = transformer.transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.Float64, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_integer_cast_transformer_transform_exclude_columns(frame_integer: pl.DataFrame) -> None:
    transformer = IntegerCast(columns=None, dtype=pl.Int32, exclude_columns=["col2", "col5"])
    out = transformer.transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int32, "col2": pl.Float64, "col3": pl.Int32, "col4": pl.String},
        ),
    )


def test_integer_cast_transformer_transform_strict_false(frame_integer: pl.DataFrame) -> None:
    transformer = IntegerCast(columns=None, dtype=pl.Float32, strict=False)
    out = transformer.transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float64, "col3": pl.Float32, "col4": pl.String},
        ),
    )


def test_integer_cast_transformer_transform_missing_policy_ignore(
    frame_integer: pl.DataFrame,
) -> None:
    transformer = IntegerCast(
        columns=["col1", "col2", "col5"], dtype=pl.Float32, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float64,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )


def test_integer_cast_transformer_transform_missing_policy_raise(
    frame_integer: pl.DataFrame,
) -> None:
    transformer = IntegerCast(columns=["col1", "col3", "col5"], dtype=pl.Int32)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_integer)


def test_integer_cast_transformer_transform_missing_policy_warn(
    frame_integer: pl.DataFrame,
) -> None:
    transformer = IntegerCast(
        columns=["col1", "col2", "col5"], dtype=pl.Float32, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_integer)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float64,
                "col3": pl.Int64,
                "col4": pl.String,
            },
        ),
    )
