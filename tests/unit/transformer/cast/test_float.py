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
from grizz.transformer import FloatCast, InplaceFloatCast


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.Float64, "col4": pl.String},
    )


##########################################
#     Tests for FloatCastTransformer     #
##########################################


def test_float_cast_transformer_repr() -> None:
    assert repr(FloatCast(columns=["col1", "col3"], prefix="", suffix="_out", dtype=pl.Int32)) == (
        "FloatCastTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out', dtype=Int32)"
    )


def test_float_cast_transformer_str() -> None:
    assert str(FloatCast(columns=["col1", "col3"], prefix="", suffix="_out", dtype=pl.Int32)) == (
        "FloatCastTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out', dtype=Int32)"
    )


def test_float_cast_transformer_equal_true() -> None:
    assert FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out")
    )


def test_float_cast_transformer_equal_false_different_columns() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(columns=["col1", "col2", "col3"], dtype=pl.Int32, prefix="", suffix="_out")
    )


def test_float_cast_transformer_equal_false_different_dtype() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(columns=["col1", "col3"], dtype=pl.Int64, prefix="", suffix="_out")
    )


def test_float_cast_transformer_equal_false_different_prefix() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="bin_", suffix="_out")
    )


def test_float_cast_transformer_equal_false_different_suffix() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="")
    )


def test_float_cast_transformer_equal_false_different_exclude_columns() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(
            columns=["col1", "col3"],
            dtype=pl.Int32,
            prefix="",
            suffix="_out",
            exclude_columns=["col4"],
        )
    )


def test_float_cast_transformer_equal_false_different_exist_policy() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(
            columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out", exist_policy="warn"
        )
    )


def test_float_cast_transformer_equal_false_different_missing_policy() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(
            columns=["col1", "col3"],
            dtype=pl.Int32,
            prefix="",
            suffix="_out",
            missing_policy="warn",
        )
    )


def test_float_cast_transformer_equal_false_different_kwargs() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out", strict=True)
    )


def test_float_cast_transformer_equal_false_different_type() -> None:
    assert not FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out").equal(
        42
    )


def test_float_cast_transformer_get_args() -> None:
    assert objects_are_equal(
        FloatCast(
            columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out", strict=True
        ).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "dtype": pl.Int32,
            "strict": True,
        },
    )


def test_float_cast_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'FloatCastTransformer.fit' as there are no parameters available to fit"
    )


def test_float_cast_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = FloatCast(
        columns=["col1", "col3", "col5"],
        dtype=pl.Float32,
        prefix="",
        suffix="_out",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_float_cast_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(
        columns=["col1", "col3", "col5"], dtype=pl.Float32, prefix="", suffix="_out"
    )
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_float_cast_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = FloatCast(
        columns=["col1", "col3", "col5"],
        dtype=pl.Float32,
        prefix="",
        suffix="_out",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_float_cast_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
                "col3_out": [1, 2, 3, 4, 5],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Float64,
                "col4": pl.String,
                "col3_out": pl.Int32,
            },
        ),
    )


def test_float_cast_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="_out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
                "col3_out": [1, 2, 3, 4, 5],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Float64,
                "col4": pl.String,
                "col3_out": pl.Int32,
            },
        ),
    )


def test_float_cast_transformer_transform_float32(dataframe: pl.DataFrame) -> None:
    transformer = FloatCast(columns=["col1", "col3"], dtype=pl.Float32, prefix="", suffix="_out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
                "col3_out": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Float64,
                "col4": pl.String,
                "col3_out": pl.Float32,
            },
        ),
    )


def test_float_cast_transformer_transform_exclude_columns(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(
        columns=None, dtype=pl.Int32, prefix="", suffix="_out", exclude_columns=["col2", "col4"]
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
                "col3_out": [1, 2, 3, 4, 5],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Float64,
                "col4": pl.String,
                "col3_out": pl.Int32,
            },
        ),
    )


def test_float_cast_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(
        columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Int32,
                "col4": pl.String,
            },
        ),
    )


def test_float_cast_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match=r"2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_float_cast_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(
        columns=["col1", "col3"], dtype=pl.Int32, prefix="", suffix="", exist_policy="warn"
    )
    with pytest.warns(
        ColumnExistsWarning,
        match=r"2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Int32,
                "col4": pl.String,
            },
        ),
    )


def test_float_cast_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(
        columns=["col1", "col3", "col5"],
        dtype=pl.Int32,
        prefix="",
        suffix="_out",
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
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
                "col3_out": [1, 2, 3, 4, 5],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Float64,
                "col4": pl.String,
                "col3_out": pl.Int32,
            },
        ),
    )


def test_float_cast_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(
        columns=["col1", "col3", "col5"], dtype=pl.Int32, prefix="", suffix="_out"
    )
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_float_cast_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FloatCast(
        columns=["col1", "col3", "col5"],
        dtype=pl.Int32,
        prefix="",
        suffix="_out",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
                "col3_out": [1, 2, 3, 4, 5],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.Float64,
                "col4": pl.String,
                "col3_out": pl.Int32,
            },
        ),
    )


#################################################
#     Tests for InplaceFloatCastTransformer     #
#################################################


def test_inplace_float_cast_transformer_repr() -> None:
    assert repr(InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "InplaceFloatCastTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', dtype=Int32)"
    )


def test_inplace_float_cast_transformer_str() -> None:
    assert str(InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "InplaceFloatCastTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', dtype=Int32)"
    )


def test_inplace_float_cast_transformer_equal_true() -> None:
    assert InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32).equal(
        InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32)
    )


def test_inplace_float_cast_transformer_equal_false_different_columns() -> None:
    assert not InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32).equal(
        InplaceFloatCast(columns=["col1", "col2", "col3"], dtype=pl.Int32)
    )


def test_inplace_float_cast_transformer_equal_false_different_dtype() -> None:
    assert not InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32).equal(
        InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int64)
    )


def test_inplace_float_cast_transformer_equal_false_different_exclude_columns() -> None:
    assert not InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32).equal(
        InplaceFloatCast(
            columns=["col1", "col3"],
            dtype=pl.Int32,
            exclude_columns=["col4"],
        )
    )


def test_inplace_float_cast_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32).equal(
        InplaceFloatCast(
            columns=["col1", "col3"],
            dtype=pl.Int32,
            missing_policy="warn",
        )
    )


def test_inplace_float_cast_transformer_equal_false_different_kwargs() -> None:
    assert not InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32).equal(
        InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32, strict=True)
    )


def test_inplace_float_cast_transformer_equal_false_different_type() -> None:
    assert not InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32).equal(42)


def test_inplace_float_cast_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32, strict=True).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "dtype": pl.Int32,
            "strict": True,
        },
    )


def test_inplace_float_cast_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'InplaceFloatCastTransformer.fit' as there are no parameters available to fit"
    )


def test_inplace_float_cast_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFloatCast(
        columns=["col1", "col3", "col5"],
        dtype=pl.Float32,
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_inplace_float_cast_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFloatCast(columns=["col1", "col3", "col5"], dtype=pl.Float32)
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_inplace_float_cast_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFloatCast(
        columns=["col1", "col3", "col5"],
        dtype=pl.Float32,
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_inplace_float_cast_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32)
    out = transformer.fit_transform(dataframe)
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


def test_inplace_float_cast_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Int32)
    out = transformer.transform(dataframe)
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


def test_inplace_float_cast_transformer_transform_float32(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFloatCast(columns=["col1", "col3"], dtype=pl.Float32)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.Float32, "col4": pl.String},
        ),
    )


def test_inplace_float_cast_transformer_transform_exclude_columns(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFloatCast(columns=None, dtype=pl.Int32, exclude_columns=["col2", "col4"])
    out = transformer.transform(dataframe)
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


def test_inplace_float_cast_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFloatCast(
        columns=["col1", "col3", "col5"], dtype=pl.Int32, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
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


def test_inplace_float_cast_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFloatCast(columns=["col1", "col3", "col5"], dtype=pl.Int32)
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_inplace_float_cast_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFloatCast(
        columns=["col1", "col3", "col5"], dtype=pl.Int32, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
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
