from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import ToDatetime

###########################################
#     Tests for ToDatetimeTransformer     #
###########################################


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


def test_to_datetime_transformer_repr() -> None:
    assert repr(ToDatetime(columns=["col1", "col3"])) == (
        "ToDatetimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_to_datetime_transformer_repr_with_kwargs() -> None:
    assert repr(ToDatetime(columns=["col1", "col3"], strict=False)) == (
        "ToDatetimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_to_datetime_transformer_str() -> None:
    assert str(ToDatetime(columns=["col1", "col3"])) == (
        "ToDatetimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_to_datetime_transformer_str_with_kwargs() -> None:
    assert str(ToDatetime(columns=["col1", "col3"], strict=False)) == (
        "ToDatetimeTransformer(columns=('col1', 'col3'), format=None, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_to_datetime_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ToDatetime(columns=["col1", "col3"])
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ToDatetimeTransformer.fit' as there are no parameters available to fit"
    )


def test_to_datetime_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ToDatetime(columns=["col1", "col3"])
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


def test_to_datetime_transformer_transform_no_format(dataframe: pl.DataFrame) -> None:
    transformer = ToDatetime(columns=["col1", "col3"])
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


def test_to_datetime_transformer_transform_format(dataframe: pl.DataFrame) -> None:
    transformer = ToDatetime(columns=["col1", "col3"], format="%Y-%m-%d %H:%M:%S", time_zone="UTC")
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


def test_to_datetime_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = ToDatetime(
        exclude_columns=["col2", "col4", "col5"], format="%Y-%m-%d %H:%M:%S", time_zone="UTC"
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


def test_to_datetime_transformer_transform_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = ToDatetime(columns=["col1", "col3", "col5"], missing_policy="ignore")
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


def test_to_datetime_transformer_transform_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = ToDatetime(columns=["col1", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_to_datetime_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ToDatetime(columns=["col1", "col3", "col5"], missing_policy="warn")
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
