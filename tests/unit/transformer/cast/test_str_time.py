from __future__ import annotations

import datetime
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
from grizz.transformer import InplaceStringToTime, StringToTime

#############################################
#     Tests for StringToTimeTransformer     #
#############################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )


def test_string_to_time_transformer_repr() -> None:
    assert repr(StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out')"
    )


def test_string_to_time_transformer_repr_with_kwargs() -> None:
    assert repr(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S")
    ) == (
        "StringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out', format='%H:%M:%S')"
    )


def test_string_to_time_transformer_str() -> None:
    assert str(StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out')"
    )


def test_string_to_time_transformer_str_with_kwargs() -> None:
    assert str(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S")
    ) == (
        "StringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', "
        "missing_policy='raise', prefix='', suffix='_out', format='%H:%M:%S')"
    )


def test_string_to_time_transformer_equal_true() -> None:
    assert StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    )


def test_string_to_time_transformer_equal_false_different_columns() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


def test_string_to_time_transformer_equal_false_different_prefix() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="bin_", suffix="_out")
    )


def test_string_to_time_transformer_equal_false_different_suffix() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="")
    )


def test_string_to_time_transformer_equal_false_different_exclude_columns() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


def test_string_to_time_transformer_equal_false_different_exist_policy() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


def test_string_to_time_transformer_equal_false_different_missing_policy() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


def test_string_to_time_transformer_equal_false_different_propagate_nulls() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out", propagate_nulls=False)
    )


def test_string_to_time_transformer_equal_false_different_kwargs() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StringToTime(columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S")
    )


def test_string_to_time_transformer_equal_false_different_type() -> None:
    assert not StringToTime(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


def test_string_to_time_transformer_get_args() -> None:
    assert objects_are_equal(
        StringToTime(
            columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S"
        ).get_args(),
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


def test_string_to_time_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    transformer.fit(dataframe)


def test_string_to_time_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_string_to_time_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_string_to_time_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_string_to_time_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
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


def test_string_to_time_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.transform(dataframe)
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


def test_string_to_time_transformer_transform_format(dataframe: pl.DataFrame) -> None:
    transformer = StringToTime(
        columns=["col1", "col3"], prefix="", suffix="_out", format="%H:%M:%S"
    )
    out = transformer.transform(dataframe)
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


def test_string_to_time_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = StringToTime(
        columns=None, prefix="", suffix="_out", exclude_columns=["col4", "col2"]
    )
    out = transformer.transform(dataframe)
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


def test_string_to_time_transformer_transform_nulls() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", None, "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [None, "12:12:12", "13:13:13", "08:08:08", None],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")
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


def test_string_to_time_transformer_transform_time() -> None:
    transformer = StringToTime(columns=["col1", "col3"], prefix="", suffix="_out")
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
                "col3_out": pl.Time,
            },
        ),
    )


def test_string_to_time_transformer_transform_incompatible_columns() -> None:
    transformer = StringToTime(columns=None, prefix="", suffix="_out")
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.Int32, "col3": pl.String, "col4": pl.String},
    )
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
                "col2": [1, 2, 3, 4, 5],
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
                "col4_out": [
                    datetime.time(hour=1, minute=1, second=1),
                    datetime.time(hour=2, minute=2, second=2),
                    datetime.time(hour=12, minute=00, second=1),
                    datetime.time(hour=18, minute=18, second=18),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Int32,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.Time,
                "col3_out": pl.Time,
                "col4_out": pl.Time,
            },
        ),
    )


def test_string_to_time_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(
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


def test_string_to_time_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_string_to_time_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn")
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


def test_string_to_time_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
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


def test_string_to_time_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_string_to_time_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StringToTime(
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


####################################################
#     Tests for InplaceStringToTimeTransformer     #
####################################################


def test_inplace_string_to_time_transformer_repr() -> None:
    assert repr(InplaceStringToTime(columns=["col1", "col3"])) == (
        "InplaceStringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_string_to_time_transformer_repr_with_kwargs() -> None:
    assert repr(InplaceStringToTime(columns=["col1", "col3"], format="%H:%M:%S")) == (
        "InplaceStringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format='%H:%M:%S')"
    )


def test_inplace_string_to_time_transformer_str() -> None:
    assert str(InplaceStringToTime(columns=["col1", "col3"])) == (
        "InplaceStringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_string_to_time_transformer_str_with_kwargs() -> None:
    assert str(InplaceStringToTime(columns=["col1", "col3"], format="%H:%M:%S")) == (
        "InplaceStringToTimeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', format='%H:%M:%S')"
    )


def test_inplace_string_to_time_transformer_equal_true() -> None:
    assert InplaceStringToTime(columns=["col1", "col3"]).equal(
        InplaceStringToTime(columns=["col1", "col3"])
    )


def test_inplace_string_to_time_transformer_equal_false_different_columns() -> None:
    assert not InplaceStringToTime(columns=["col1", "col3"]).equal(
        InplaceStringToTime(columns=["col1", "col2", "col3"])
    )


def test_inplace_string_to_time_transformer_equal_false_different_exclude_columns() -> None:
    assert not InplaceStringToTime(columns=["col1", "col3"]).equal(
        InplaceStringToTime(columns=["col1", "col3"], exclude_columns=["col4"])
    )


def test_inplace_string_to_time_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceStringToTime(columns=["col1", "col3"]).equal(
        InplaceStringToTime(columns=["col1", "col3"], missing_policy="warn")
    )


def test_inplace_string_to_time_transformer_equal_false_different_propagate_nulls() -> None:
    assert not InplaceStringToTime(columns=["col1", "col3"]).equal(
        InplaceStringToTime(columns=["col1", "col3"], propagate_nulls=False)
    )


def test_inplace_string_to_time_transformer_equal_false_different_kwargs() -> None:
    assert not InplaceStringToTime(columns=["col1", "col3"]).equal(
        InplaceStringToTime(columns=["col1", "col3"], format="%H:%M:%S")
    )


def test_inplace_string_to_time_transformer_equal_false_different_type() -> None:
    assert not InplaceStringToTime(columns=["col1", "col3"]).equal(42)


def test_inplace_string_to_time_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceStringToTime(columns=["col1", "col3"], format="%H:%M:%S").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "format": "%H:%M:%S",
        },
    )


def test_inplace_string_to_time_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3"])
    transformer.fit(dataframe)


def test_inplace_string_to_time_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_inplace_string_to_time_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_inplace_string_to_time_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_inplace_string_to_time_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3"])
    out = transformer.fit_transform(dataframe)
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


def test_inplace_string_to_time_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3"])
    out = transformer.transform(dataframe)
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


def test_inplace_string_to_time_transformer_transform_format(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3"], format="%H:%M:%S")
    out = transformer.transform(dataframe)
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


def test_inplace_string_to_time_transformer_transform_exclude_columns(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToTime(columns=None, exclude_columns=["col4", "col2"])
    out = transformer.transform(dataframe)
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


def test_inplace_string_to_time_transformer_transform_nulls() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", None, "18:18:18", "23:59:59"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [None, "12:12:12", "13:13:13", "08:08:08", None],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = InplaceStringToTime(columns=["col1", "col3"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    datetime.time(1, 1, 1),
                    datetime.time(2, 2, 2),
                    None,
                    datetime.time(18, 18, 18),
                    datetime.time(23, 59, 59),
                ],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [
                    None,
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    None,
                ],
                "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            },
            schema={"col1": pl.Time, "col2": pl.String, "col3": pl.Time, "col4": pl.String},
        ),
    )


def test_inplace_string_to_time_transformer_transform_time() -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3"])
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


def test_inplace_string_to_time_transformer_transform_incompatible_columns() -> None:
    transformer = InplaceStringToTime(columns=None)
    frame = pl.DataFrame(
        {
            "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["11:11:11", "12:12:12", "13:13:13", "08:08:08", "23:59:59"],
            "col4": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
        },
        schema={"col1": pl.String, "col2": pl.Int32, "col3": pl.String, "col4": pl.String},
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
                "col2": [1, 2, 3, 4, 5],
                "col3": [
                    datetime.time(hour=11, minute=11, second=11),
                    datetime.time(hour=12, minute=12, second=12),
                    datetime.time(hour=13, minute=13, second=13),
                    datetime.time(hour=8, minute=8, second=8),
                    datetime.time(hour=23, minute=59, second=59),
                ],
                "col4": [
                    datetime.time(hour=1, minute=1, second=1),
                    datetime.time(hour=2, minute=2, second=2),
                    datetime.time(hour=12, minute=00, second=1),
                    datetime.time(hour=18, minute=18, second=18),
                    datetime.time(hour=23, minute=59, second=59),
                ],
            },
            schema={
                "col1": pl.Time,
                "col2": pl.Int32,
                "col3": pl.Time,
                "col4": pl.Time,
            },
        ),
    )


def test_inplace_string_to_time_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
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


def test_inplace_string_to_time_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_inplace_string_to_time_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStringToTime(columns=["col1", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
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
