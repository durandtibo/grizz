from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import polars as pl
import pytest
from coola import objects_are_equal
from coola.testing import numpy_available
from coola.utils import is_numpy_available
from polars.testing import assert_frame_equal

from grizz.utils.null import (
    compute_null,
    compute_null_count,
    compute_temporal_null_count,
    propagate_nulls,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_numpy_available():
    import numpy as np

##################################
#     Tests for compute_null     #
##################################


@numpy_available
def test_compute_null() -> None:
    assert_frame_equal(
        compute_null(
            pl.DataFrame(
                {
                    "int": [None, 1, 0, 1],
                    "float": [1.2, 4.2, None, 2.2],
                    "str": ["A", "B", None, None],
                },
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            )
        ),
        pl.DataFrame(
            {
                "column": ["int", "float", "str"],
                "null": [1, 1, 2],
                "total": [4, 4, 4],
                "null_pct": [0.25, 0.25, 0.5],
            },
            schema={
                "column": pl.String,
                "null": pl.Int64,
                "total": pl.Int64,
                "null_pct": pl.Float64,
            },
        ),
    )


@numpy_available
def test_compute_null_empty_row() -> None:
    assert_frame_equal(
        compute_null(
            pl.DataFrame(
                {"int": [], "float": [], "str": []},
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            )
        ),
        pl.DataFrame(
            {
                "column": ["int", "float", "str"],
                "null": [0, 0, 0],
                "total": [0, 0, 0],
                "null_pct": [float("nan"), float("nan"), float("nan")],
            },
            schema={
                "column": pl.String,
                "null": pl.Int64,
                "total": pl.Int64,
                "null_pct": pl.Float64,
            },
        ),
    )


@numpy_available
def test_compute_null_empty() -> None:
    assert_frame_equal(
        compute_null(pl.DataFrame({})),
        pl.DataFrame(
            {
                "column": [],
                "null": [],
                "total": [],
                "null_pct": [],
            },
            schema={
                "column": pl.String,
                "null": pl.Int64,
                "total": pl.Int64,
                "null_pct": pl.Float64,
            },
        ),
    )


########################################
#     Tests for compute_null_count     #
########################################


@numpy_available
def test_compute_null_count() -> None:
    assert objects_are_equal(
        compute_null_count(
            pl.DataFrame(
                {
                    "int": [None, 1, 0, 1],
                    "float": [1.2, 4.2, None, 2.2],
                    "str": ["A", "B", None, None],
                },
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            )
        ),
        np.array([1, 1, 2], dtype=np.int64),
    )


@numpy_available
def test_compute_null_count_empty_rows() -> None:
    assert objects_are_equal(
        compute_null_count(
            pl.DataFrame(
                {"int": [], "float": [], "str": []},
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            )
        ),
        np.array([0, 0, 0], dtype=np.int64),
    )


@numpy_available
def test_compute_null_count_empty() -> None:
    assert objects_are_equal(compute_null_count(pl.DataFrame({})), np.array([], dtype=np.int64))


################################################
#    Tests for compute_temporal_null_count     #
################################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [None, float("nan"), 0.0, 1.0],
            "col2": [None, 1, 0, None],
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col1": pl.Float64,
            "col2": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


@pytest.fixture
def dataframe_empty() -> pl.DataFrame:
    return pl.DataFrame(
        {"col1": [], "col2": [], "datetime": []},
        schema={
            "col1": pl.Float64,
            "col2": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


@numpy_available
@pytest.mark.parametrize("columns", [["col1", "col2"], ("col1", "col2")])
def test_compute_temporal_null_count(dataframe: pl.DataFrame, columns: Sequence[str]) -> None:
    assert objects_are_equal(
        compute_temporal_null_count(
            frame=dataframe, columns=columns, dt_column="datetime", period="1mo"
        ),
        (
            np.array([2, 0, 0, 1], dtype=np.int64),
            np.array([2, 2, 2, 2], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


@numpy_available
def test_compute_temporal_null_count_subset(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null_count(
            frame=dataframe, columns=["col1"], dt_column="datetime", period="1mo"
        ),
        (
            np.array([1, 0, 0, 0], dtype=np.int64),
            np.array([1, 1, 1, 1], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


@numpy_available
def test_compute_temporal_null_count_empty_columns(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null_count(
            frame=dataframe, columns=[], dt_column="datetime", period="1mo"
        ),
        (
            np.array([0, 0, 0, 0], dtype=np.int64),
            np.array([0, 0, 0, 0], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


@numpy_available
def test_compute_temporal_null_count_empty(dataframe_empty: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null_count(
            frame=dataframe_empty,
            columns=["col1", "col2"],
            dt_column="datetime",
            period="1mo",
        ),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), []),
    )


@numpy_available
def test_compute_temporal_null_count_monthly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null_count(
            frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="1mo"
        ),
        (
            np.array([2, 0, 0, 1], dtype=np.int64),
            np.array([2, 2, 2, 2], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


@numpy_available
def test_compute_temporal_null_count_biweekly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null_count(
            frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="2w"
        ),
        (
            np.array([2, 0, 0, 1], dtype=np.int64),
            np.array([2, 2, 2, 2], dtype=np.int64),
            ["2019 week 52", "2020 week 04", "2020 week 08", "2020 week 12"],
        ),
    )


#####################################
#     Tests for propagate_nulls     #
#####################################


def test_propagate_nulls() -> None:
    assert_frame_equal(
        propagate_nulls(
            frame=pl.DataFrame(
                {
                    "col1": [1.0, 2.0, 3.0, 4.0],
                    "col2": ["a", "b", "c", "d"],
                    "col3": [1, 2, 3, 4],
                    "col4": [1, 2, 3, 4],
                },
                schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.Int64, "col4": pl.Int32},
            ),
            frame_with_null=pl.DataFrame(
                {
                    "col1": [0.0, 0.0, None, float("nan")],
                    "col2": ["meow", None, "miaou", None],
                    "col3": [None, 1, 1, None],
                    "col4": [0, 0, 0, 0],
                }
            ),
        ),
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, None, 4.0],
                "col2": ["a", None, "c", None],
                "col3": [None, 2, 3, None],
                "col4": [1, 2, 3, 4],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.Int64, "col4": pl.Int32},
        ),
    )


def test_propagate_nulls_empty() -> None:
    assert_frame_equal(
        propagate_nulls(
            frame=pl.DataFrame({"col1": [], "col2": [], "col3": []}),
            frame_with_null=pl.DataFrame({"col1": [], "col2": [], "col3": []}),
        ),
        pl.DataFrame({"col1": [], "col2": [], "col3": []}),
    )
