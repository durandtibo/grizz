from __future__ import annotations

import datetime
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
from grizz.transformer import TimeToSecond, ToTime

#############################################
#     Tests for TimeToSecondTransformer     #
#############################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
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


def test_time_to_second_transformer_repr() -> None:
    assert repr(TimeToSecond(in_col="time", out_col="second")).startswith(
        "TimeToSecondTransformer("
    )


def test_time_to_second_transformer_str() -> None:
    assert str(TimeToSecond(in_col="time", out_col="second")).startswith("TimeToSecondTransformer(")


def test_time_to_second_transformer_equal_true() -> None:
    assert TimeToSecond(in_col="col1", out_col="out").equal(
        TimeToSecond(in_col="col1", out_col="out")
    )


def test_time_to_second_transformer_equal_false_different_in_col() -> None:
    assert not TimeToSecond(in_col="col1", out_col="out").equal(
        TimeToSecond(in_col="col3", out_col="out")
    )


def test_time_to_second_transformer_equal_false_different_out_col() -> None:
    assert not TimeToSecond(in_col="col1", out_col="out").equal(
        TimeToSecond(in_col="col1", out_col="out2")
    )


def test_time_to_second_transformer_equal_false_different_exist_policy() -> None:
    assert not TimeToSecond(in_col="col1", out_col="out").equal(
        TimeToSecond(in_col="col1", out_col="out", exist_policy="warn")
    )


def test_time_to_second_transformer_equal_false_different_missing_policy() -> None:
    assert not TimeToSecond(in_col="col1", out_col="out").equal(
        TimeToSecond(in_col="col1", out_col="out", missing_policy="warn")
    )


def test_time_to_second_transformer_equal_false_different_type() -> None:
    assert not TimeToSecond(in_col="col1", out_col="out").equal(42)


def test_time_to_second_transformer_get_args() -> None:
    assert objects_are_equal(
        TimeToSecond(in_col="col1", out_col="out").get_args(),
        {
            "in_col": "col1",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_time_to_second_transformer_fit(
    caplog: pytest.LogCaptureFixture, dataframe: pl.DataFrame
) -> None:
    transformer = TimeToSecond(in_col="time", out_col="second")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'TimeToSecondTransformer.fit' as there are no parameters available to fit"
    )


def test_time_to_second_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_time_to_second_transformer_fit_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second")
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_time_to_second_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_time_to_second_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="time", out_col="second")
    out = transformer.fit_transform(dataframe)
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


def test_time_to_second_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="time", out_col="second")
    out = transformer.transform(dataframe)
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


def test_time_to_second_transformer_transform_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="time", out_col="col", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
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
                "col": [1.89, 61.89, 3661.89, 1159.89, 69559.42],
            },
            schema={"time": pl.Time, "col": pl.Float64},
        ),
    )


def test_time_to_second_transformer_transform_exist_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="time", out_col="col")
    with pytest.raises(ColumnExistsError, match="column 'col' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_time_to_second_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="time", out_col="col", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
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
                "col": [1.89, 61.89, 3661.89, 1159.89, 69559.42],
            },
            schema={"time": pl.Time, "col": pl.Float64},
        ),
    )


def test_time_to_second_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
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


def test_time_to_second_transformer_transform_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second")
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_time_to_second_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
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
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format=None)"
    )


def test_to_time_transformer_repr_with_kwargs() -> None:
    assert repr(ToTime(columns=["col1", "col3"], strict=False)) == (
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format=None, strict=False)"
    )


def test_to_time_transformer_str() -> None:
    assert str(ToTime(columns=["col1", "col3"])) == (
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format=None)"
    )


def test_to_time_transformer_str_with_kwargs() -> None:
    assert str(ToTime(columns=["col1", "col3"], strict=False)) == (
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format=None, strict=False)"
    )


def test_to_time_transformer_equal_true() -> None:
    assert ToTime(columns=["col1", "col3"]).equal(ToTime(columns=["col1", "col3"]))


def test_to_time_transformer_equal_false_different_columns() -> None:
    assert not ToTime(columns=["col1", "col3"]).equal(ToTime(columns=["col1", "col2", "col3"]))


def test_to_time_transformer_equal_false_different_exclude_columns() -> None:
    assert not ToTime(columns=["col1", "col3"]).equal(
        ToTime(columns=["col1", "col3"], exclude_columns=["col2"])
    )


def test_to_time_transformer_equal_false_different_missing_policy() -> None:
    assert not ToTime(columns=["col1", "col3"]).equal(
        ToTime(columns=["col1", "col3"], missing_policy="warn")
    )


def test_to_time_transformer_equal_false_different_format() -> None:
    assert not ToTime(columns=["col1", "col3"]).equal(
        ToTime(columns=["col1", "col3"], format="%H:%M:%S")
    )


def test_to_time_transformer_equal_false_different_kwargs() -> None:
    assert not ToTime(columns=["col1", "col3"]).equal(ToTime(columns=["col1", "col3"], strict=True))


def test_to_time_transformer_equal_false_different_type() -> None:
    assert not ToTime(columns=["col1", "col3"]).equal(42)


def test_to_time_transformer_get_args() -> None:
    assert objects_are_equal(
        ToTime(columns=["col1", "col2", "col3"], strict=True).get_args(),
        {
            "columns": ("col1", "col2", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "format": None,
            "strict": True,
        },
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


def test_to_time_transformer_fit_missing_policy_ignore(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(
        columns=["col1", "col3", "col5"], format="%H:%M:%S", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(frame_time)


def test_to_time_transformer_fit_missing_policy_raise(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3", "col5"], format="%H:%M:%S")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(frame_time)


def test_to_time_transformer_fit_missing_policy_warn(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3", "col5"], format="%H:%M:%S", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(frame_time)


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
