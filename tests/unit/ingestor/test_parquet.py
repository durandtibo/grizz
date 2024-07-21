from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.ingestor import ParquetIngestor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("frame.parquet")
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    frame.write_parquet(path)
    return path


#####################################
#     Tests for ParquetIngestor     #
#####################################


def test_parquet_ingestor_repr(frame_path: Path) -> None:
    assert repr(ParquetIngestor(frame_path)).startswith("ParquetIngestor(")


def test_parquet_ingestor_repr_with_kwargs(frame_path: Path) -> None:
    assert repr(ParquetIngestor(frame_path, columns=["col1", "col3"])).startswith(
        "ParquetIngestor("
    )


def test_parquet_ingestor_str(frame_path: Path) -> None:
    assert str(ParquetIngestor(frame_path)).startswith("ParquetIngestor(")


def test_parquet_ingestor_str_with_kwargs(frame_path: Path) -> None:
    assert str(ParquetIngestor(frame_path, columns=["col1", "col3"])).startswith("ParquetIngestor(")


def test_parquet_ingestor_ingest(frame_path: Path) -> None:
    assert_frame_equal(
        ParquetIngestor(frame_path).ingest(),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_parquet_ingestor_ingest_with_kwargs(frame_path: Path) -> None:
    assert_frame_equal(
        ParquetIngestor(frame_path, columns=["col1", "col3"]).ingest(),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
