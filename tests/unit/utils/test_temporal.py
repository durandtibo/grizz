from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.utils.temporal import (
    compute_temporal_stats,
    to_step_names,
    to_temporal_frames,
)

############################################
#     Tests for compute_temporal_stats     #
############################################


def test_compute_temporal_stats() -> None:
    stats = compute_temporal_stats(
        pl.DataFrame(
            {
                "col": list(range(104)),
                "datetime": [datetime(year=2020, month=1, day=3, tzinfo=timezone.utc)] * 101
                + [
                    datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                ],
            },
            schema={
                "col": pl.Int64,
                "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        ),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert_frame_equal(
        stats,
        pl.DataFrame(
            {
                "step": [
                    datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2020, month=2, day=1, tzinfo=timezone.utc),
                    datetime(year=2020, month=3, day=1, tzinfo=timezone.utc),
                    datetime(year=2020, month=4, day=1, tzinfo=timezone.utc),
                ],
                "count": [101, 1, 1, 1],
                "nunique": [101, 1, 1, 1],
                "mean": [50.0, 101.0, 102.0, 103.0],
                "std": [29.300170647967224, None, None, None],
                "min": [0.0, 101.0, 102.0, 103.0],
                "q01": [1.0, 101.0, 102.0, 103.0],
                "q05": [5.0, 101.0, 102.0, 103.0],
                "q10": [10.0, 101.0, 102.0, 103.0],
                "q25": [25.0, 101.0, 102.0, 103.0],
                "median": [50.0, 101.0, 102.0, 103.0],
                "q75": [75.0, 101.0, 102.0, 103.0],
                "q90": [90.0, 101.0, 102.0, 103.0],
                "q95": [95.0, 101.0, 102.0, 103.0],
                "q99": [99.0, 101.0, 102.0, 103.0],
                "max": [100.0, 101.0, 102.0, 103.0],
            },
            schema={
                "step": pl.Datetime(time_unit="us", time_zone="UTC"),
                "count": pl.Int64,
                "nunique": pl.Int64,
                "mean": pl.Float64,
                "std": pl.Float64,
                "min": pl.Float64,
                "q01": pl.Float64,
                "q05": pl.Float64,
                "q10": pl.Float64,
                "q25": pl.Float64,
                "median": pl.Float64,
                "q75": pl.Float64,
                "q90": pl.Float64,
                "q95": pl.Float64,
                "q99": pl.Float64,
                "max": pl.Float64,
            },
        ),
    )


def test_compute_temporal_stats_empty() -> None:
    out = compute_temporal_stats(
        pl.DataFrame(
            {"col": [], "datetime": []},
            schema={
                "col": pl.Float64,
                "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        ),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "step": [],
                "count": [],
                "nunique": [],
                "mean": [],
                "std": [],
                "min": [],
                "q01": [],
                "q05": [],
                "q10": [],
                "q25": [],
                "median": [],
                "q75": [],
                "q90": [],
                "q95": [],
                "q99": [],
                "max": [],
            },
            schema={
                "step": pl.Datetime(time_unit="us", time_zone="UTC"),
                "count": pl.Int64,
                "nunique": pl.Int64,
                "mean": pl.Float64,
                "std": pl.Float64,
                "min": pl.Float64,
                "q01": pl.Float64,
                "q05": pl.Float64,
                "q10": pl.Float64,
                "q25": pl.Float64,
                "median": pl.Float64,
                "q75": pl.Float64,
                "q90": pl.Float64,
                "q95": pl.Float64,
                "q99": pl.Float64,
                "max": pl.Float64,
            },
        ),
    )


########################################
#     Tests for to_temporal_frames     #
########################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "datetime": [
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col": pl.Float64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


def test_to_temporal_frames_monthly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_temporal_frames(dataframe, dt_column="datetime", period="1mo"),
        (
            [
                pl.DataFrame(
                    {
                        "col": [1.0, 2.0, 3.0],
                        "datetime": [
                            datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                            datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                            datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                        ],
                    },
                    schema={
                        "col": pl.Float64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
                pl.DataFrame(
                    {
                        "col": [4.0],
                        "datetime": [datetime(year=2020, month=2, day=3, tzinfo=timezone.utc)],
                    },
                    schema={
                        "col": pl.Float64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
                pl.DataFrame(
                    {
                        "col": [5.0],
                        "datetime": [datetime(year=2020, month=3, day=3, tzinfo=timezone.utc)],
                    },
                    schema={
                        "col": pl.Float64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
                pl.DataFrame(
                    {
                        "col": [0.0],
                        "datetime": [datetime(year=2020, month=4, day=3, tzinfo=timezone.utc)],
                    },
                    schema={
                        "col": pl.Float64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
            ],
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_to_temporal_frames_yearly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_temporal_frames(dataframe, dt_column="datetime", period="1y"),
        (
            [
                pl.DataFrame(
                    {
                        "col": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
                        "datetime": [
                            datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                            datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                            datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                            datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                            datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                            datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                        ],
                    },
                    schema={
                        "col": pl.Float64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                )
            ],
            ["2020"],
        ),
    )


def test_to_temporal_frames_empty() -> None:
    assert objects_are_equal(
        to_temporal_frames(pl.DataFrame({}), dt_column="datetime", period="1mo"), ([], [])
    )


###################################
#     Tests for to_step_names     #
###################################


def test_to_step_names_monthly(dataframe: pl.DataFrame) -> None:
    groups = dataframe.sort("datetime").group_by_dynamic("datetime", every="1mo")
    assert objects_are_equal(
        to_step_names(groups=groups, period="1mo"), ["2020-01", "2020-02", "2020-03", "2020-04"]
    )


def test_to_step_names_yearly(dataframe: pl.DataFrame) -> None:
    groups = dataframe.sort("datetime").group_by_dynamic("datetime", every="1y")
    assert objects_are_equal(to_step_names(groups=groups, period="1y"), ["2020"])


def test_to_step_names_empty() -> None:
    groups = (
        pl.DataFrame(
            {"col": [], "datetime": []},
            schema={
                "col": pl.Float64,
                "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        )
        .sort("datetime")
        .group_by_dynamic("datetime", every="1mo")
    )
    assert objects_are_equal(to_step_names(groups=groups, period="1mo"), [])
