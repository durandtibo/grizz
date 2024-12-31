from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from grizz.utils.count import (
    compute_nunique,
    compute_temporal_count,
    compute_temporal_value_counts,
)

if is_numpy_available():
    import numpy as np

####################################
#    Tests for compute_nunique     #
####################################


@numpy_available
def test_compute_nunique() -> None:
    assert objects_are_equal(
        compute_nunique(
            frame=pl.DataFrame(
                {
                    "int": [None, 1, 0, 1],
                    "float": [1.2, 4.2, None, 2.2],
                    "str": ["A", "B", None, None],
                },
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            ),
        ),
        np.array([3, 4, 3], dtype=np.int64),
    )


@numpy_available
def test_compute_nunique_empty_rows() -> None:
    assert objects_are_equal(
        compute_nunique(
            frame=pl.DataFrame(
                {"int": [], "float": [], "str": []},
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            ),
        ),
        np.array([0, 0, 0], dtype=np.int64),
    )


@numpy_available
def test_compute_nunique_empty() -> None:
    assert objects_are_equal(compute_nunique(frame=pl.DataFrame({})), np.array([], dtype=np.int64))


###########################################
#    Tests for compute_temporal_count     #
###########################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ]
        },
        schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


@numpy_available
def test_compute_temporal_count_month(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_count(
            frame=dataframe,
            dt_column="datetime",
            period="1mo",
        ),
        (
            np.array([3, 1, 1, 1], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


@numpy_available
def test_compute_temporal_count_biweekly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_count(
            frame=dataframe,
            dt_column="datetime",
            period="2w",
        ),
        (
            np.array([3, 1, 1, 1], dtype=np.int64),
            ["2019 week 52", "2020 week 04", "2020 week 08", "2020 week 12"],
        ),
    )


@numpy_available
def test_compute_temporal_count_empty() -> None:
    assert objects_are_equal(
        compute_temporal_count(
            frame=pl.DataFrame(
                {"datetime": []},
                schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
            ),
            dt_column="datetime",
            period="1mo",
        ),
        (np.array([], dtype=np.int64), []),
    )


##################################################
#    Tests for compute_temporal_value_counts     #
##################################################


@numpy_available
def test_compute_temporal_value_counts() -> None:
    counts, steps, values = compute_temporal_value_counts(
        pl.DataFrame(
            {
                "col": [None, 1.0, 0.0, 1.0, 1.0, 4.2, 42.0],
                "datetime": [
                    datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                    datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                ],
            },
            schema={
                "col": pl.Float64,
                "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        ),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_equal(
        counts, np.array([[1, 0, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    )
    assert objects_are_equal(steps, ["2020-01", "2020-02", "2020-03", "2020-04"])
    assert objects_are_equal(values, ["0.0", "1.0", "4.2", "42.0", "null"])


@numpy_available
def test_compute_temporal_value_counts_drop_nulls() -> None:
    counts, steps, values = compute_temporal_value_counts(
        pl.DataFrame(
            {
                "col": [None, 1.0, 0.0, 1.0, 1.0, 4.2, 42.0],
                "datetime": [
                    datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                    datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                ],
            },
            schema={
                "col": pl.Float64,
                "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        ),
        column="col",
        dt_column="datetime",
        period="1mo",
        drop_nulls=True,
    )
    assert objects_are_equal(
        counts, np.array([[1, 0, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    assert objects_are_equal(steps, ["2020-01", "2020-02", "2020-03", "2020-04"])
    assert objects_are_equal(values, ["0.0", "1.0", "4.2", "42.0"])


@numpy_available
def test_compute_temporal_value_counts_empty() -> None:
    counts, steps, values = compute_temporal_value_counts(
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
    assert objects_are_equal(counts, np.zeros((0, 0), dtype=np.int64))
    assert objects_are_equal(steps, [])
    assert objects_are_equal(values, [])
