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
from grizz.transformer import TimeToSecond

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
    with pytest.raises(ColumnNotFoundError, match=r"column 'in' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_time_to_second_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'in' is missing in the DataFrame and will be ignored"
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
    with pytest.raises(ColumnExistsError, match=r"column 'col' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_time_to_second_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="time", out_col="col", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match=r"column 'col' already exists in the DataFrame and will be overwritten",
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
    with pytest.raises(ColumnNotFoundError, match=r"column 'in' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_time_to_second_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = TimeToSecond(in_col="in", out_col="second", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'in' is missing in the DataFrame and will be ignored"
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
