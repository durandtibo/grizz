from __future__ import annotations

import logging
from datetime import time

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import Cast, ToTime


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


#####################################
#     Tests for CastTransformer     #
#####################################


def test_cast_transformer_repr() -> None:
    assert repr(Cast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)"
    )


def test_cast_transformer_repr_with_kwargs() -> None:
    assert repr(Cast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False, "
        "strict=False)"
    )


def test_cast_transformer_str() -> None:
    assert str(Cast(columns=["col1", "col3"], dtype=pl.Int32)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)"
    )


def test_cast_transformer_str_with_kwargs() -> None:
    assert str(Cast(columns=["col1", "col3"], dtype=pl.Int32, strict=False)) == (
        "CastTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False, "
        "strict=False)"
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


def test_cast_transformer_transform_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Cast(columns=["col1", "col3", "col5"], dtype=pl.Float32)
    with pytest.raises(RuntimeError, match="column col5 is not in the DataFrame"):
        transformer.transform(dataframe)


def test_cast_transformer_transform_ignore_missing_true(
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


#######################################
#     Tests for ToTimeTransformer     #
#######################################


def test_to_time_transformer_repr() -> None:
    assert repr(ToTime(columns=["col1", "col3"])) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, ignore_missing=False)"
    )


def test_to_time_transformer_repr_with_kwargs() -> None:
    assert repr(ToTime(columns=["col1", "col3"], strict=False)) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, ignore_missing=False, "
        "strict=False)"
    )


def test_to_time_transformer_str() -> None:
    assert str(ToTime(columns=["col1", "col3"])) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, ignore_missing=False)"
    )


def test_to_time_transformer_str_with_kwargs() -> None:
    assert str(ToTime(columns=["col1", "col3"], strict=False)) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, ignore_missing=False, "
        "strict=False)"
    )


def test_to_time_transformer_transform_no_format() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = ToTime(columns=["col1", "col3"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    time(1, 1, 1),
                    time(2, 2, 2),
                    time(12, 0, 1),
                    time(18, 18, 18),
                    time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    time(11, 11, 11),
                    time(12, 12, 12),
                    time(13, 13, 13),
                    time(8, 8, 8),
                    time(23, 59, 59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_transformer_transform_format() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = ToTime(columns=["col1", "col3"], format="%H:%M:%S")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    time(1, 1, 1),
                    time(2, 2, 2),
                    time(12, 0, 1),
                    time(18, 18, 18),
                    time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    time(11, 11, 11),
                    time(12, 12, 12),
                    time(13, 13, 13),
                    time(8, 8, 8),
                    time(23, 59, 59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_transformer_transform_ignore_missing_false() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = ToTime(columns=["col1", "col3", "col5"], format="%H:%M:%S")
    with pytest.raises(RuntimeError, match="column col5 is not in the DataFrame"):
        transformer.transform(frame)


def test_to_time_transformer_transform_ignore_missing_true(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = ToTime(columns=["col1", "col3", "col5"], format="%H:%M:%S", ignore_missing=True)
    with caplog.at_level(logging.WARNING):
        out = transformer.transform(frame)
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "col1": [
                        time(1, 1, 1),
                        time(2, 2, 2),
                        time(12, 0, 1),
                        time(18, 18, 18),
                        time(23, 59, 59),
                    ],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": [
                        time(11, 11, 11),
                        time(12, 12, 12),
                        time(13, 13, 13),
                        time(8, 8, 8),
                        time(23, 59, 59),
                    ],
                    "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                },
                schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
            ),
        )
        assert caplog.messages[-1].startswith(
            "skipping transformation for column col5 because the column is missing"
        )
