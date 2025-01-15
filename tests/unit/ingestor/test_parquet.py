from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import DataFrameNotFoundError
from grizz.ingestor import ParquetFileIngestor

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


#########################################
#     Tests for ParquetFileIngestor     #
#########################################


def test_parquet_file_ingestor_repr(frame_path: Path) -> None:
    assert repr(ParquetFileIngestor(frame_path)).startswith("ParquetFileIngestor(")


def test_parquet_file_ingestor_repr_with_kwargs(frame_path: Path) -> None:
    assert repr(ParquetFileIngestor(frame_path, columns=["col1", "col3"])).startswith(
        "ParquetFileIngestor("
    )


def test_parquet_file_ingestor_str(frame_path: Path) -> None:
    assert str(ParquetFileIngestor(frame_path)).startswith("ParquetFileIngestor(")


def test_parquet_file_ingestor_str_with_kwargs(frame_path: Path) -> None:
    assert str(ParquetFileIngestor(frame_path, columns=["col1", "col3"])).startswith(
        "ParquetFileIngestor("
    )


def test_parquet_file_ingestor_equal_true(tmp_path: Path) -> None:
    assert ParquetFileIngestor(tmp_path.joinpath("data.parquet")).equal(
        ParquetFileIngestor(tmp_path.joinpath("data.parquet"))
    )


def test_parquet_file_ingestor_equal_false_different_path(tmp_path: Path) -> None:
    assert not ParquetFileIngestor(tmp_path.joinpath("data.parquet")).equal(
        ParquetFileIngestor(tmp_path.joinpath("data2.parquet"))
    )


def test_parquet_file_ingestor_equal_false_different_kwargs(tmp_path: Path) -> None:
    assert not ParquetFileIngestor(tmp_path.joinpath("data.parquet")).equal(
        ParquetFileIngestor(tmp_path.joinpath("data.parquet"), include_header=False)
    )


def test_parquet_file_ingestor_equal_false_different_type(tmp_path: Path) -> None:
    assert not ParquetFileIngestor(tmp_path.joinpath("data.parquet")).equal(42)


def test_parquet_file_ingestor_ingest(frame_path: Path) -> None:
    assert_frame_equal(
        ParquetFileIngestor(frame_path).ingest(),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_parquet_file_ingestor_ingest_with_kwargs(frame_path: Path) -> None:
    assert_frame_equal(
        ParquetFileIngestor(frame_path, columns=["col1", "col3"]).ingest(),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_parquet_file_ingestor_ingest_missing_path(tmp_path: Path) -> None:
    ingestor = ParquetFileIngestor(tmp_path.joinpath("data.parquet"))
    with pytest.raises(DataFrameNotFoundError, match="DataFrame file does not exist"):
        ingestor.ingest()
