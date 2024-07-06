from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.ingestor import CsvIngestor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("frame.csv")
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    frame.write_csv(path)
    return path


#################################
#     Tests for CsvIngestor     #
#################################


def test_csv_ingestor_repr(frame_path: Path) -> None:
    assert repr(CsvIngestor(frame_path)).startswith("CsvIngestor(")


def test_csv_ingestor_str(frame_path: Path) -> None:
    assert str(CsvIngestor(frame_path)).startswith("CsvIngestor(")


def test_csv_ingestor_str_with_kwargs(frame_path: Path) -> None:
    assert str(CsvIngestor(frame_path, usecols=["col1", "col3"])).startswith("CsvIngestor(")


def test_csv_ingestor_ingest(frame_path: Path) -> None:
    assert_frame_equal(
        CsvIngestor(frame_path).ingest(),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_csv_ingestor_ingest_with_kwargs(frame_path: Path) -> None:
    assert_frame_equal(
        CsvIngestor(frame_path, columns=["col1", "col3"]).ingest(),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
