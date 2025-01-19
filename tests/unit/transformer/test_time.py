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
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )


def test_to_time_transformer_repr() -> None:
    assert repr(ToTime(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out')"
    )


def test_to_time_transformer_repr_with_kwargs() -> None:
    assert repr(ToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S")) == (
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out', format='%H:%M:%S')"
    )


def test_to_time_transformer_str() -> None:
    assert str(ToTime(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out')"
    )


def test_to_time_transformer_str_with_kwargs() -> None:
    assert str(ToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S")) == (
        "ToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out', format='%H:%M:%S')"
    )


def test_to_time_transformer_equal_true() -> None:
    assert ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    )


def test_to_time_transformer_equal_false_different_columns() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


def test_to_time_transformer_equal_false_different_prefix() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="bin_", suffix="_out")
    )


def test_to_time_transformer_equal_false_different_suffix() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="")
    )


def test_to_time_transformer_equal_false_different_exclude_columns() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


def test_to_time_transformer_equal_false_different_exist_policy() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


def test_to_time_transformer_equal_false_different_missing_policy() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


def test_to_time_transformer_equal_false_different_propagate_nulls() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="_out", propagate_nulls=False)
    )


def test_to_time_transformer_equal_false_different_kwargs() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S")
    )


def test_to_time_transformer_equal_false_different_type() -> None:
    assert not ToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


def test_to_time_transformer_get_args() -> None:
    assert objects_are_equal(
        ToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "format": "%H:%M:%S",
        },
    )


def test_standard_scaler_transformer_fit(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    transformer.fit(frame_time)


def test_standard_scaler_transformer_fit_missing_policy_ignore(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(frame_time)


def test_standard_scaler_transformer_fit_missing_policy_raise(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(frame_time)


def test_standard_scaler_transformer_fit_missing_policy_warn(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(frame_time)


def test_standard_scaler_transformer_fit_transform(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )


def test_standard_scaler_transformer_transform(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )


def test_to_time_transformer_transform_format(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S")
    out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )


def test_to_time_transformer_transform_exclude_columns(frame_time: pl.DataFrame) -> None:
    transformer = ToTime(columns=None, prefix="", suffix="_out", exclude_columns=["col4", "col2"])
    out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )


def test_standard_scaler_transformer_transform_nulls() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", None, "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [None, "12:12:12", "13:13:13", "08:08:08", None],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", None, "18:18:18", "23:59:59"],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [None, "12:12:12", "13:13:13", "08:08:08", None],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    None,
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    None,
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    None,
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )


def test_standard_scaler_transformer_transform_time() -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    frame = pl.DataFrame(
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
    out = transformer.transform(frame)
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
                "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.Time,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )


def test_standard_scaler_transformer_transform_incompatible_columns() -> None:
    transformer = ToTime(columns=None, prefix="", suffix="_out")
    frame = pl.DataFrame(
        {
            "col1": [
                datetime.time(1, 1, 1),
                datetime.time(2, 2, 2),
                datetime.time(12, 0, 1),
                datetime.time(18, 18, 18),
                datetime.time(23, 59, 59),
            ],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.Time, "col2": pl.Int32, "col3": pl.String, "col4": pl.String},
    )
    with pytest.raises(pl.exceptions.SchemaError):
        transformer.transform(frame)


def test_standard_scaler_transformer_transform_exist_policy_ignore(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_standard_scaler_transformer_transform_exist_policy_raise(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(frame_time)


def test_standard_scaler_transformer_transform_exist_policy_warn(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
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


def test_standard_scaler_transformer_transform_missing_policy_ignore(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )


def test_standard_scaler_transformer_transform_missing_policy_raise(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_time)


def test_standard_scaler_transformer_transform_missing_policy_warn(
    frame_time: pl.DataFrame,
) -> None:
    transformer = ToTime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_time)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col1_out": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    datetime.time(12, 0, 1),
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col3_out": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
            },
        ),
    )
