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
from grizz.transformer import Diff, TimeDiff


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
        schema={"col1": pl.Int64, "col2": pl.String},
    )


#####################################
#     Tests for DiffTransformer     #
#####################################


def test_diff_transformer_repr() -> None:
    assert str(Diff(in_col="col1", out_col="diff")).startswith("DiffTransformer(")


def test_diff_transformer_str() -> None:
    assert str(Diff(in_col="col1", out_col="diff")).startswith("DiffTransformer(")


def test_diff_transformer_equal_true() -> None:
    assert Diff(in_col="col4", out_col="out").equal(Diff(in_col="col4", out_col="out"))


def test_diff_transformer_equal_false_different_in_col() -> None:
    assert not Diff(in_col="col4", out_col="out").equal(Diff(in_col="col3", out_col="out"))


def test_diff_transformer_equal_false_different_out_col() -> None:
    assert not Diff(in_col="col4", out_col="out").equal(Diff(in_col="col4", out_col="out2"))


def test_diff_transformer_equal_false_different_exist_policy() -> None:
    assert not Diff(in_col="col4", out_col="out").equal(
        Diff(in_col="col4", out_col="out", exist_policy="warn")
    )


def test_diff_transformer_equal_false_different_missing_policy() -> None:
    assert not Diff(in_col="col4", out_col="out").equal(
        Diff(in_col="col4", out_col="out", missing_policy="warn")
    )


def test_diff_transformer_equal_false_different_shift() -> None:
    assert not Diff(in_col="col4", out_col="out").equal(Diff(in_col="col4", out_col="out", shift=2))


def test_diff_transformer_equal_false_different_type() -> None:
    assert not Diff(in_col="col4", out_col="out").equal(42)


def test_diff_transformer_get_args() -> None:
    assert objects_are_equal(
        Diff(in_col="col4", out_col="out", shift=1).get_args(),
        {
            "in_col": "col4",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
            "shift": 1,
        },
    )


def test_diff_transformer_fit(dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    transformer = Diff(in_col="col1", out_col="diff")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'DiffTransformer.fit' as there are no parameters available to fit"
    )


def test_diff_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = Diff(in_col="in", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_diff_transformer_fit_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = Diff(in_col="in", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_diff_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = Diff(in_col="in", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_diff_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Diff(in_col="col1", out_col="diff")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "diff": [None, 1, 1, 1, 1],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "diff": pl.Int64},
        ),
    )


def test_diff_transformer_transform_int32(dataframe: pl.DataFrame) -> None:
    transformer = Diff(in_col="col1", out_col="diff")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "diff": [None, 1, 1, 1, 1],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "diff": pl.Int64},
        ),
    )


def test_diff_transformer_transform_float32() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Float32, "col2": pl.String},
    )
    transformer = Diff(in_col="col1", out_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["a", "b", "c", "d", "e"],
                "diff": [None, 1.0, 1.0, 1.0, 1.0],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "diff": pl.Float32},
        ),
    )


def test_diff_transformer_transform_shift_2(dataframe: pl.DataFrame) -> None:
    transformer = Diff(in_col="col1", out_col="diff", shift=2)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "diff": [None, None, 2, 2, 2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "diff": pl.Int64},
        ),
    )


def test_diff_transformer_transform_empty() -> None:
    frame = pl.DataFrame(
        {"col1": [], "col2": []},
        schema={"col1": pl.Int64, "col2": pl.String},
    )
    transformer = Diff(in_col="col1", out_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [], "col2": [], "diff": []},
            schema={"col1": pl.Int64, "col2": pl.String, "diff": pl.Int64},
        ),
    )


def test_diff_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Diff(in_col="col1", out_col="col2", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [None, 1, 1, 1, 1]},
            schema={"col1": pl.Int64, "col2": pl.Int64},
        ),
    )


def test_diff_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Diff(in_col="col1", out_col="col2")
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_diff_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Diff(in_col="col1", out_col="col2", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [None, 1, 1, 1, 1]},
            schema={"col1": pl.Int64, "col2": pl.Int64},
        ),
    )


def test_diff_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Diff(in_col="in", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            schema={"col1": pl.Int64, "col2": pl.String},
        ),
    )


def test_diff_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Diff(in_col="in", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_diff_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Diff(in_col="in", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            schema={"col1": pl.Int64, "col2": pl.String},
        ),
    )


#########################################
#     Tests for TimeDiffTransformer     #
#########################################


def test_time_diff_transformer_repr() -> None:
    assert repr(TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")).startswith(
        "TimeDiffTransformer("
    )


def test_time_diff_transformer_str() -> None:
    assert str(TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")).startswith(
        "TimeDiffTransformer("
    )


def test_time_diff_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'TimeDiffTransformer.fit' as there are no parameters available to fit"
    )


def test_time_diff_transformer_fit_transform() -> None:
    frame = pl.DataFrame(
        {
            "col": ["b", "b", "b", "c", "a", "a", "a", "b", "c", "d"],
            "time": [8, 2, 3, 4, 5, 6, 7, 1, 9, 10],
        },
        schema={"col": pl.String, "time": pl.Int64},
    )
    transformer = TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")
    out = transformer.fit_transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col": ["a", "a", "a", "b", "b", "b", "b", "c", "c", "d"],
                "time": [5, 6, 7, 1, 2, 3, 8, 4, 9, 10],
                "diff": [0, 1, 1, 0, 1, 1, 5, 0, 5, 0],
            },
            schema={"col": pl.String, "time": pl.Int64, "diff": pl.Int64},
        ),
    )


def test_time_diff_transformer_transform() -> None:
    frame = pl.DataFrame(
        {
            "col": ["b", "b", "b", "c", "a", "a", "a", "b", "c", "d"],
            "time": [8, 2, 3, 4, 5, 6, 7, 1, 9, 10],
        },
        schema={"col": pl.String, "time": pl.Int64},
    )
    transformer = TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col": ["a", "a", "a", "b", "b", "b", "b", "c", "c", "d"],
                "time": [5, 6, 7, 1, 2, 3, 8, 4, 9, 10],
                "diff": [0, 1, 1, 0, 1, 1, 5, 0, 5, 0],
            },
            schema={"col": pl.String, "time": pl.Int64, "diff": pl.Int64},
        ),
    )


def test_time_diff_transformer_transform_int64() -> None:
    frame = pl.DataFrame(
        {
            "col": ["a", "b", "a", "a", "b"],
            "time": [1, 2, 3, 4, 5],
        },
        schema={"col": pl.String, "time": pl.Int64},
    )
    transformer = TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col": ["a", "a", "a", "b", "b"],
                "time": [1, 3, 4, 2, 5],
                "diff": [0, 2, 1, 0, 3],
            },
            schema={"col": pl.String, "time": pl.Int64, "diff": pl.Int64},
        ),
    )


def test_time_diff_transformer_transform_float64() -> None:
    frame = pl.DataFrame(
        {
            "col": ["a", "b", "a", "a", "b"],
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        schema={"col": pl.String, "time": pl.Float64},
    )
    transformer = TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col": ["a", "a", "a", "b", "b"],
                "time": [1.0, 3.0, 4.0, 2.0, 5.0],
                "diff": [0.0, 2.0, 1.0, 0.0, 3.0],
            },
            schema={"col": pl.String, "time": pl.Float64, "diff": pl.Float64},
        ),
    )


def test_time_diff_transformer_transform_empty() -> None:
    frame = pl.DataFrame(
        {"col": [], "time": []},
        schema={"col": pl.String, "time": pl.Int64},
    )
    transformer = TimeDiff(group_cols=["col"], time_col="time", time_diff_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col": [], "time": [], "diff": []},
            schema={"col": pl.String, "time": pl.Int64, "diff": pl.Int64},
        ),
    )
