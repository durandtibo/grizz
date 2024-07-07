from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import Cast


@pytest.fixture()
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


##############################################
#     Tests for CastDataFrameTransformer     #
##############################################


def test_cast_dataframe_transformer_repr() -> None:
    assert repr(Cast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "CastDataFrameTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)"
    )


def test_cast_dataframe_transformer_repr_with_kwargs() -> None:
    assert repr(Cast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "CastDataFrameTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False, "
        "strict=False)"
    )


def test_cast_dataframe_transformer_str() -> None:
    assert str(Cast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "CastDataFrameTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)"
    )


def test_cast_dataframe_transformer_str_with_kwargs() -> None:
    assert str(Cast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "CastDataFrameTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False, "
        "strict=False)"
    )


def test_cast_dataframe_transformer_transform_int32(dataframe: pl.DataFrame) -> None:
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


def test_cast_dataframe_transformer_transform_float32(dataframe: pl.DataFrame) -> None:
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


def test_cast_dataframe_transformer_transform_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Cast(columns=["col1", "col3", "col5"], dtype=pl.Float32)
    with pytest.raises(RuntimeError, match="column col5 is not in the DataFrame"):
        transformer.transform(dataframe)


def test_cast_dataframe_transformer_transform_ignore_missing_true(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = Cast(columns=["col1", "col3", "col5"], dtype=pl.Float32, ignore_missing=True)
    with caplog.at_level(logging.WARNING):
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
        assert caplog.messages[-1].startswith(
            "skipping transformation for column col5 because the column is missing"
        )
