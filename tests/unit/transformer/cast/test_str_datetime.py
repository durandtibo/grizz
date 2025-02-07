from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone

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
from grizz.transformer import InplaceStringToDatetime, StringToDatetime


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=1,
                    minute=1,
                    second=1,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=2,
                    minute=2,
                    second=2,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=12,
                    minute=0,
                    second=1,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=18,
                    minute=18,
                    second=18,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=23,
                    minute=59,
                    second=59,
                    tzinfo=timezone(timedelta(0)),
                ),
            ],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [
                "2020-01-01 11:11:11",
                "2020-02-01 12:12:12",
                "2020-03-01 13:13:13",
                "2020-04-01 08:08:08",
                "2020-05-01 23:59:59",
            ],
            "col4": [
                "2020-01-01 01:01:01",
                "2020-01-01 02:02:02",
                "2020-01-01 12:00:01",
                "2020-01-01 18:18:18",
                "2020-01-01 23:59:59",
            ],
        },
        schema={
            "col1": pl.Datetime(time_unit="us"),
            "col2": pl.String,
            "col3": pl.String,
            "col4": pl.String,
        },
    )


#################################################
#     Tests for StringToDatetimeTransformer     #
#################################################


def test_string_to_datetime_transformer_repr() -> None:
    assert repr(StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_string_to_datetime_transformer_repr_with_kwargs() -> None:
    assert repr(
        StringToDatetime(
            columns=["col1", "col3"], prefix="", suffix="_out", format="%Y-%m-%d %H:%M:%S"
        )
    ) == (
        "StringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', "
        "format='%Y-%m-%d %H:%M:%S')"
    )


def test_string_to_datetime_transformer_str() -> None:
    assert str(StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_string_to_datetime_transformer_str_with_kwargs() -> None:
    assert str(
        StringToDatetime(
            columns=["col1", "col3"], prefix="", suffix="_out", format="%Y-%m-%d %H:%M:%S"
        )
    ) == (
        "StringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', "
        "format='%Y-%m-%d %H:%M:%S')"
    )


def test_string_to_datetime_transformer_equal_true() -> None:
    assert StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out")
    )


def test_string_to_datetime_transformer_equal_false_different_columns() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


def test_string_to_datetime_transformer_equal_false_different_prefix() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(columns=["col1", "col3"], prefix="bin_", suffix="_out")
    )


def test_string_to_datetime_transformer_equal_false_different_suffix() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(columns=["col1", "col3"], prefix="", suffix="")
    )


def test_string_to_datetime_transformer_equal_false_different_exclude_columns() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(
            columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col2"]
        )
    )


def test_string_to_datetime_transformer_equal_false_different_exist_policy() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


def test_string_to_datetime_transformer_equal_false_different_missing_policy() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


def test_string_to_datetime_transformer_equal_false_different_missing_kwargs() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToDatetime(
            columns=["col1", "col3"], prefix="", suffix="_out", format="%Y-%m-%d %H:%M:%S"
        )
    )


def test_string_to_datetime_transformer_equal_false_different_type() -> None:
    assert not StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


def test_string_to_datetime_transformer_get_args() -> None:
    assert objects_are_equal(
        StringToDatetime(
            columns=["col1", "col3"], prefix="", suffix="_out", format="%Y-%m-%d %H:%M:%S"
        ).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "format": "%Y-%m-%d %H:%M:%S",
        },
    )


def test_string_to_datetime_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'StringToDatetimeTransformer.fit' as there are no parameters available to fit"
    )


def test_string_to_datetime_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = StringToDatetime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_string_to_datetime_transformer_fit_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = StringToDatetime(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_string_to_datetime_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = StringToDatetime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_string_to_datetime_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    "2020-01-01 11:11:11",
                    "2020-02-01 12:12:12",
                    "2020-03-01 13:13:13",
                    "2020-04-01 08:08:08",
                    "2020-05-01 23:59:59",
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
                "col3_out": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col3_out": pl.Datetime(time_unit="us"),
            },
        ),
    )


def test_string_to_datetime_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = StringToDatetime(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    "2020-01-01 11:11:11",
                    "2020-02-01 12:12:12",
                    "2020-03-01 13:13:13",
                    "2020-04-01 08:08:08",
                    "2020-05-01 23:59:59",
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
                "col3_out": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col3_out": pl.Datetime(time_unit="us"),
            },
        ),
    )


def test_string_to_datetime_transformer_transform_format(dataframe: pl.DataFrame) -> None:
    transformer = StringToDatetime(
        columns=["col1", "col3"],
        prefix="",
        suffix="_out",
        format="%Y-%m-%d %H:%M:%S",
        time_zone="UTC",
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    "2020-01-01 11:11:11",
                    "2020-02-01 12:12:12",
                    "2020-03-01 13:13:13",
                    "2020-04-01 08:08:08",
                    "2020-05-01 23:59:59",
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
                "col3_out": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone.utc,
                    ),
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col3_out": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        ),
    )


def test_string_to_datetime_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = StringToDatetime(
        columns=None,
        exclude_columns=["col2", "col4", "col5"],
        prefix="",
        suffix="_out",
        format="%Y-%m-%d %H:%M:%S",
        time_zone="UTC",
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    "2020-01-01 11:11:11",
                    "2020-02-01 12:12:12",
                    "2020-03-01 13:13:13",
                    "2020-04-01 08:08:08",
                    "2020-05-01 23:59:59",
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
                "col3_out": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone.utc,
                    ),
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col3_out": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        ),
    )


def test_string_to_datetime_transformer_transform_incompatible_columns() -> None:
    transformer = StringToDatetime(columns=None, prefix="", suffix="_out")
    frame = pl.DataFrame(
        {
            "col1": [
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=1,
                    minute=1,
                    second=1,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=2,
                    minute=2,
                    second=2,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=12,
                    minute=0,
                    second=1,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=18,
                    minute=18,
                    second=18,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=23,
                    minute=59,
                    second=59,
                    tzinfo=timezone(timedelta(0)),
                ),
            ],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={
            "col1": pl.Datetime(time_unit="us"),
            "col2": pl.Int32,
            "col3": pl.String,
            "col4": pl.String,
        },
    )
    with pytest.raises((pl.exceptions.SchemaError, pl.exceptions.ComputeError)):
        transformer.transform(frame)


def test_string_to_datetime_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToDatetime(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us"),
                "col4": pl.String,
            },
        ),
    )


def test_string_to_datetime_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToDatetime(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_string_to_datetime_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToDatetime(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn"
    )
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
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us"),
                "col4": pl.String,
            },
        ),
    )


def test_string_to_datetime_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToDatetime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    "2020-01-01 11:11:11",
                    "2020-02-01 12:12:12",
                    "2020-03-01 13:13:13",
                    "2020-04-01 08:08:08",
                    "2020-05-01 23:59:59",
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
                "col3_out": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col3_out": pl.Datetime(time_unit="us"),
            },
        ),
    )


def test_string_to_datetime_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToDatetime(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_string_to_datetime_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToDatetime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    "2020-01-01 11:11:11",
                    "2020-02-01 12:12:12",
                    "2020-03-01 13:13:13",
                    "2020-04-01 08:08:08",
                    "2020-05-01 23:59:59",
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
                "col3_out": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "col3_out": pl.Datetime(time_unit="us"),
            },
        ),
    )


########################################################
#     Tests for InplaceStringToDatetimeTransformer     #
########################################################


def test_inplace_string_to_datetime_transformer_repr() -> None:
    assert repr(InplaceStringToDatetime(columns=["col1", "col3"])) == (
        "InplaceStringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_string_to_datetime_transformer_repr_with_kwargs() -> None:
    assert repr(InplaceStringToDatetime(columns=["col1", "col3"], format="%Y-%m-%d %H:%M:%S")) == (
        "InplaceStringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format='%Y-%m-%d %H:%M:%S')"
    )


def test_inplace_string_to_datetime_transformer_str() -> None:
    assert str(InplaceStringToDatetime(columns=["col1", "col3"])) == (
        "InplaceStringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_string_to_datetime_transformer_str_with_kwargs() -> None:
    assert str(InplaceStringToDatetime(columns=["col1", "col3"], format="%Y-%m-%d %H:%M:%S")) == (
        "InplaceStringToDatetimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format='%Y-%m-%d %H:%M:%S')"
    )


def test_inplace_string_to_datetime_transformer_equal_true() -> None:
    assert InplaceStringToDatetime(columns=["col1", "col3"]).equal(
        InplaceStringToDatetime(columns=["col1", "col3"])
    )


def test_inplace_string_to_datetime_transformer_equal_false_different_columns() -> None:
    assert not InplaceStringToDatetime(columns=["col1", "col3"]).equal(
        InplaceStringToDatetime(columns=["col1", "col2", "col3"])
    )


def test_inplace_string_to_datetime_transformer_equal_false_different_exclude_columns() -> None:
    assert not InplaceStringToDatetime(columns=["col1", "col3"]).equal(
        InplaceStringToDatetime(columns=["col1", "col3"], exclude_columns=["col2"])
    )


def test_inplace_string_to_datetime_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceStringToDatetime(columns=["col1", "col3"]).equal(
        InplaceStringToDatetime(columns=["col1", "col3"], missing_policy="warn")
    )


def test_inplace_string_to_datetime_transformer_equal_false_different_missing_kwargs() -> None:
    assert not InplaceStringToDatetime(columns=["col1", "col3"]).equal(
        InplaceStringToDatetime(columns=["col1", "col3"], format="%Y-%m-%d %H:%M:%S")
    )


def test_inplace_string_to_datetime_transformer_equal_false_different_type() -> None:
    assert not InplaceStringToDatetime(columns=["col1", "col3"]).equal(42)


def test_inplace_string_to_datetime_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceStringToDatetime(columns=["col1", "col3"], format="%Y-%m-%d %H:%M:%S").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "format": "%Y-%m-%d %H:%M:%S",
        },
    )


def test_inplace_string_to_datetime_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3"])
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'InplaceStringToDatetimeTransformer.fit' as there are no parameters available to fit"
    )


def test_inplace_string_to_datetime_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_inplace_string_to_datetime_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_inplace_string_to_datetime_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_inplace_string_to_datetime_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3"])
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us"),
                "col4": pl.String,
            },
        ),
    )


def test_inplace_string_to_datetime_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us"),
                "col4": pl.String,
            },
        ),
    )


def test_inplace_string_to_datetime_transformer_transform_format(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStringToDatetime(
        columns=["col1", "col3"],
        format="%Y-%m-%d %H:%M:%S",
        time_zone="UTC",
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone.utc,
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us", time_zone="UTC"),
                "col4": pl.String,
            },
        ),
    )


def test_inplace_string_to_datetime_transformer_transform_exclude_columns(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToDatetime(
        columns=None,
        exclude_columns=["col2", "col4", "col5"],
        format="%Y-%m-%d %H:%M:%S",
        time_zone="UTC",
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone.utc,
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone.utc,
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us", time_zone="UTC"),
                "col4": pl.String,
            },
        ),
    )


def test_inplace_string_to_datetime_transformer_transform_incompatible_columns() -> None:
    transformer = InplaceStringToDatetime(columns=None)
    frame = pl.DataFrame(
        {
            "col1": [
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=1,
                    minute=1,
                    second=1,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=2,
                    minute=2,
                    second=2,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=12,
                    minute=0,
                    second=1,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=18,
                    minute=18,
                    second=18,
                    tzinfo=timezone(timedelta(0)),
                ),
                datetime(
                    year=2020,
                    month=1,
                    day=1,
                    hour=23,
                    minute=59,
                    second=59,
                    tzinfo=timezone(timedelta(0)),
                ),
            ],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={
            "col1": pl.Datetime(time_unit="us"),
            "col2": pl.Int32,
            "col3": pl.String,
            "col4": pl.String,
        },
    )
    with pytest.raises((pl.exceptions.SchemaError, pl.exceptions.ComputeError)):
        transformer.transform(frame)


def test_inplace_string_to_datetime_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us"),
                "col4": pl.String,
            },
        ),
    )


def test_inplace_string_to_datetime_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_inplace_string_to_datetime_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToDatetime(columns=["col1", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=1,
                        minute=1,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=2,
                        minute=2,
                        second=2,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=12,
                        minute=0,
                        second=1,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=18,
                        minute=18,
                        second=18,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    datetime(
                        year=2020,
                        month=1,
                        day=1,
                        hour=11,
                        minute=11,
                        second=11,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=2,
                        day=1,
                        hour=12,
                        minute=12,
                        second=12,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=3,
                        day=1,
                        hour=13,
                        minute=13,
                        second=13,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=4,
                        day=1,
                        hour=8,
                        minute=8,
                        second=8,
                        tzinfo=timezone(timedelta(0)),
                    ),
                    datetime(
                        year=2020,
                        month=5,
                        day=1,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=timezone(timedelta(0)),
                    ),
                ],
                "col4": [
                    "2020-01-01 01:01:01",
                    "2020-01-01 02:02:02",
                    "2020-01-01 12:00:01",
                    "2020-01-01 18:18:18",
                    "2020-01-01 23:59:59",
                ],
            },
            schema={
                "col1": pl.Datetime(time_unit="us"),
                "col2": pl.String,
                "col3": pl.Datetime(time_unit="us"),
                "col4": pl.String,
            },
        ),
    )
