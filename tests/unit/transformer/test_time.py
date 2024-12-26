from __future__ import annotations

import datetime
import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import TimeToSecond, ToTime

#############################################
#     Tests for TimeToSecondTransformer     #
#############################################


def test_time_to_second_transformer_repr() -> None:
    assert repr(TimeToSecond(in_col="time", out_col="second")).startswith(
        "TimeToSecondTransformer("
    )


def test_time_to_second_transformer_str() -> None:
    assert str(TimeToSecond(in_col="time", out_col="second")).startswith("TimeToSecondTransformer(")


def test_time_to_second_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = TimeToSecond(in_col="time", out_col="second")
    with caplog.at_level(logging.INFO):
        transformer.fit(
            pl.DataFrame(
                {
                    "time": [
                        datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                        datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                        datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                        datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                        datetime.time(hour=19, minute=19, second=19, microsecond=420000),
                    ],
                    "col": ["a", "b", "c", "d", "e"],
                },
                schema={"time": pl.Time, "col": pl.String},
            )
        )
    assert caplog.messages[0].startswith(
        "Skipping 'TimeToSecondTransformer.fit' as there are no parameters available to fit"
    )


def test_time_to_second_transformer_fit_transform() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                datetime.time(hour=19, minute=19, second=19, microsecond=420000),
            ],
            "col": ["a", "b", "c", "d", "e"],
        },
        schema={"time": pl.Time, "col": pl.String},
    )
    transformer = TimeToSecond(in_col="time", out_col="second")
    out = transformer.fit_transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "time": [
                    datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                    datetime.time(hour=19, minute=19, second=19, microsecond=420000),
                ],
                "col": ["a", "b", "c", "d", "e"],
                "second": [1.89, 61.89, 3661.89, 1159.89, 69559.42],
            },
            schema={"time": pl.Time, "col": pl.String, "second": pl.Float64},
        ),
    )


def test_time_to_second_transformer_transform() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                datetime.time(hour=19, minute=19, second=19, microsecond=420000),
            ],
            "col": ["a", "b", "c", "d", "e"],
        },
        schema={"time": pl.Time, "col": pl.String},
    )
    transformer = TimeToSecond(in_col="time", out_col="second")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "time": [
                    datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                    datetime.time(hour=19, minute=19, second=19, microsecond=420000),
                ],
                "col": ["a", "b", "c", "d", "e"],
                "second": [1.89, 61.89, 3661.89, 1159.89, 69559.42],
            },
            schema={"time": pl.Time, "col": pl.String, "second": pl.Float64},
        ),
    )


def test_time_to_second_transformer_transform_missing_policy_ignore() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                datetime.time(hour=19, minute=19, second=19, microsecond=420000),
            ],
            "col": ["a", "b", "c", "d", "e"],
        },
        schema={"time": pl.Time, "col": pl.String},
    )
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "time": [
                    datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                    datetime.time(hour=19, minute=19, second=19, microsecond=420000),
                ],
                "col": ["a", "b", "c", "d", "e"],
            },
            schema={"time": pl.Time, "col": pl.String},
        ),
    )


def test_time_to_second_transformer_transform_missing_policy_raise() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                datetime.time(hour=19, minute=19, second=19, microsecond=420000),
            ],
            "col": ["a", "b", "c", "d", "e"],
        },
        schema={"time": pl.Time, "col": pl.String},
    )
    transformer = TimeToSecond(in_col="in", out_col="second")
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.transform(frame)


def test_time_to_second_transformer_transform_missing_policy_warn() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                datetime.time(hour=19, minute=19, second=19, microsecond=420000),
            ],
            "col": ["a", "b", "c", "d", "e"],
        },
        schema={"time": pl.Time, "col": pl.String},
    )
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "time": [
                    datetime.time(hour=0, minute=0, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=1, minute=1, second=1, microsecond=890000),
                    datetime.time(hour=0, minute=19, second=19, microsecond=890000),
                    datetime.time(hour=19, minute=19, second=19, microsecond=420000),
                ],
                "col": ["a", "b", "c", "d", "e"],
            },
            schema={"time": pl.Time, "col": pl.String},
        ),
    )


#######################################
#     Tests for ToTimeTransformer     #
#######################################


@pytest.fixture
def frame_time() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [
                datetime.time(1, 1, 1),
                datetime.time(2, 2, 2),
                datetime.time(12, 0, 1),
                datetime.time(18, 18, 18),
                datetime.time(23, 59, 59),
            ],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.Time, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )


def test_to_time_transformer_repr() -> None:
    assert repr(ToTime(columns=["col1", "col3"])) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_to_time_transformer_repr_with_kwargs() -> None:
    assert repr(ToTime(columns=["col1", "col3"], strict=False)) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_to_time_transformer_str() -> None:
    assert str(ToTime(columns=["col1", "col3"])) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_to_time_transformer_str_with_kwargs() -> None:
    assert str(ToTime(columns=["col1", "col3"], strict=False)) == (
        "ToTimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_to_time_transformer_fit(
    frame_time: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ToTime(columns=["col1", "col3"])
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_time)
    assert caplog.messages[0].startswith(
        "Skipping 'ToTimeTransformer.fit' as there are no parameters available to fit"
    )


def test_to_time_transformer_fit_transform(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3"])
    out = transformer.fit_transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime.time(11, 11, 11),
                    datetime.time(12, 12, 12),
                    datetime.time(13, 13, 13),
                    datetime.time(8, 8, 8),
                    datetime.time(23, 59, 59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_transformer_transform_no_format(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3"])
    out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime.time(11, 11, 11),
                    datetime.time(12, 12, 12),
                    datetime.time(13, 13, 13),
                    datetime.time(8, 8, 8),
                    datetime.time(23, 59, 59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_transformer_transform_format(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3"], format="%H:%M:%S")
    out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime.time(hour=1, minute=1, second=1),
                    datetime.time(hour=2, minute=2, second=2),
                    datetime.time(hour=12, minute=0, second=1),
                    datetime.time(hour=18, minute=18, second=18),
                    datetime.time(hour=23, minute=59, second=59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_transformer_transform_exclude_columns(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3", "col4"], exclude_columns=["col4", "col5"])
    out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime.time(11, 11, 11),
                    datetime.time(12, 12, 12),
                    datetime.time(13, 13, 13),
                    datetime.time(8, 8, 8),
                    datetime.time(23, 59, 59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_transformer_transform_missing_policy_ignore(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(
        columns=["col1", "col3", "col5"], format="%H:%M:%S", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime.time(hour=1, minute=1, second=1),
                    datetime.time(hour=2, minute=2, second=2),
                    datetime.time(hour=12, minute=0, second=1),
                    datetime.time(hour=18, minute=18, second=18),
                    datetime.time(hour=23, minute=59, second=59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_to_time_transformer_transform_missing_policy_raise(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3", "col5"], format="%H:%M:%S")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_time)


def test_to_time_transformer_transform_missing_policy_warn(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3", "col5"], format="%H:%M:%S", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime.time(hour=1, minute=1, second=1),
                    datetime.time(hour=2, minute=2, second=2),
                    datetime.time(hour=12, minute=0, second=1),
                    datetime.time(hour=18, minute=18, second=18),
                    datetime.time(hour=23, minute=59, second=59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )
