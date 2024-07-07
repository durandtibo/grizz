from __future__ import annotations

import datetime

import polars as pl
from polars.testing import assert_frame_equal

from grizz.transformer import TimeToSecond

#############################################
#     Tests for TimeToSecondTransformer     #
#############################################


def test_time_to_second_dataframe_transformer_str() -> None:
    assert str(TimeToSecond(in_col="time", out_col="second")).startswith("TimeToSecondTransformer(")


def test_time_to_second_dataframe_transformer_transform() -> None:
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
