from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.lazy.ingestor import Ingestor


@pytest.fixture
def lazyframe() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )


##############################
#     Tests for Ingestor     #
##############################


def test_ingestor_repr(lazyframe: pl.LazyFrame) -> None:
    assert repr(Ingestor(frame=lazyframe)).startswith("Ingestor(")


def test_ingestor_str(lazyframe: pl.LazyFrame) -> None:
    assert str(Ingestor(frame=lazyframe)).startswith("Ingestor(")


def test_ingestor_equal_true(lazyframe: pl.LazyFrame) -> None:
    assert Ingestor(lazyframe).equal(Ingestor(lazyframe))


def test_ingestor_equal_false_different_frame(lazyframe: pl.LazyFrame) -> None:
    assert not Ingestor(lazyframe).equal(Ingestor(pl.LazyFrame()))


def test_ingestor_equal_false_different_type(lazyframe: pl.LazyFrame) -> None:
    assert not Ingestor(lazyframe).equal(42)


def test_ingestor_ingest(lazyframe: pl.LazyFrame) -> None:
    out = Ingestor(frame=lazyframe).ingest()
    assert lazyframe is out
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
