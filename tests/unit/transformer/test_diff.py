from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from grizz.transformer import Diff, TimeDiff

#####################################
#     Tests for DiffTransformer     #
#####################################


def test_diff_transformer_repr() -> None:
    assert str(Diff(in_col="col1", out_col="diff")).startswith("DiffTransformer(")


def test_diff_transformer_str() -> None:
    assert str(Diff(in_col="col1", out_col="diff")).startswith("DiffTransformer(")


def test_diff_transformer_transform_int32() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String},
    )
    transformer = Diff(in_col="col1", out_col="diff")
    out = transformer.transform(frame)
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


def test_diff_transformer_transform_shift_2() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String},
    )
    transformer = Diff(in_col="col1", out_col="diff", shift=2)
    out = transformer.transform(frame)
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
