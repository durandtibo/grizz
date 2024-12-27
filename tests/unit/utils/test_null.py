from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from grizz.utils.null import propagate_nulls

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
