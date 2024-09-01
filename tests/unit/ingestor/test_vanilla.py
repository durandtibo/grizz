from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.ingestor import Ingestor


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )


##############################
#     Tests for Ingestor     #
##############################


def test_ingestor_repr(dataframe: pl.DataFrame) -> None:
    assert repr(Ingestor(frame=dataframe)).startswith("Ingestor(")


def test_ingestor_str(dataframe: pl.DataFrame) -> None:
    assert str(Ingestor(frame=dataframe)).startswith("Ingestor(")


def test_ingestor_ingest(dataframe: pl.DataFrame) -> None:
    out = Ingestor(frame=dataframe).ingest()
    assert dataframe is out
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
