from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import DataNotFoundError
from grizz.ingestor import ParquetFileIngestor, ParquetIngestor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("frame.parquet")
    pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    ).write_parquet(path)
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


def test_parquet_ingestor_equal_true(tmp_path: Path) -> None:
    assert ParquetIngestor(tmp_path.joinpath("data.parquet")).equal(
        ParquetIngestor(tmp_path.joinpath("data.parquet"))
    )


def test_parquet_ingestor_equal_false_different_path(tmp_path: Path) -> None:
    assert not ParquetIngestor(tmp_path.joinpath("data.parquet")).equal(
        ParquetIngestor(tmp_path.joinpath("data2.parquet"))
    )


def test_parquet_ingestor_equal_false_different_kwargs(tmp_path: Path) -> None:
    assert not ParquetIngestor(tmp_path.joinpath("data.parquet")).equal(
        ParquetIngestor(tmp_path.joinpath("data.parquet"), include_header=False)
    )


def test_parquet_ingestor_equal_false_different_type(tmp_path: Path) -> None:
    assert not ParquetIngestor(tmp_path.joinpath("data.parquet")).equal(42)


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
    with pytest.raises(DataNotFoundError, match=r"Data file does not exist"):
        ingestor.ingest()
