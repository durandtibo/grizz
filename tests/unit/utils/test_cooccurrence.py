from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from grizz.utils.cooccurrence import compute_pairwise_cooccurrence

###################################################
#     Tests for compute_pairwise_cooccurrence     #
###################################################


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"col1": [0, 1, 1, 0, 0]}),
        pl.DataFrame({"col1": [False, True, False, True, False]}),
    ],
)
def test_compute_pairwise_cooccurrence_1_column(frame: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(frame),
        np.array([[2]], dtype=int),
    )


def test_compute_pairwise_cooccurrence_1_column_no_diagonal() -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(
            pl.DataFrame({"col1": [0, 1, 1, 0, 0]}), remove_diagonal=True
        ),
        np.array([[0]], dtype=int),
    )


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"col1": [0, 1, 1, 0, 0], "col2": [0, 1, 0, 1, 0]}),
        pl.DataFrame({"col1": [0, 1, 2, 0, 0], "col2": [0, 1, 0, 3, 0]}),
        pl.DataFrame(
            {
                "col1": [False, True, True, False, False],
                "col2": [False, True, False, True, False],
            }
        ),
    ],
)
def test_compute_pairwise_cooccurrence_2_columns(frame: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(frame),
        np.array([[2, 1], [1, 2]], dtype=int),
    )


def test_compute_pairwise_cooccurrence_2_columns_no_diagonal() -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(
            pl.DataFrame({"col1": [0, 1, 1, 0, 0], "col2": [0, 1, 0, 1, 0]})
        ),
        np.array([[2, 1], [1, 2]], dtype=int),
    )


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, 0],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        ),
        pl.DataFrame(
            {
                "col1": [0, 2, 1, 0, 0, 3, 0],
                "col2": [0, 5, 0, 5, 0, 5, 0],
                "col3": [0, 0, 0, 0, 6, 7, 8],
            }
        ),
        pl.DataFrame(
            {
                "col1": [False, True, True, False, False, True, False],
                "col2": [False, True, False, True, False, True, False],
                "col3": [False, False, False, False, True, True, True],
            }
        ),
    ],
)
def test_compute_pairwise_cooccurrence_3_columns(frame: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(frame),
        np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
    )


def test_compute_pairwise_cooccurrence_3_columns_no_diagonal() -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(
            pl.DataFrame(
                {
                    "col1": [0, 1, 1, 0, 0, 1, 0],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            ),
            remove_diagonal=True,
        ),
        np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=int),
    )


def test_compute_pairwise_cooccurrence_empty() -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(
            pl.DataFrame({}),
        ),
        np.zeros((0, 0), dtype=int),
    )


def test_compute_pairwise_cooccurrence_empty_rows() -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(
            pl.DataFrame({"col1": [], "col2": [], "col3": []}),
        ),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
    )


def test_compute_pairwise_cooccurrence_null() -> None:
    assert objects_are_equal(
        compute_pairwise_cooccurrence(
            pl.DataFrame(
                {
                    "col1": [0, 1, None, 0, 0, 1, 0],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, None, 1, 1],
                }
            ),
        ),
        np.array([[2, 2, 1], [2, 3, 1], [1, 1, 2]], dtype=int),
    )
