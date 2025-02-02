from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import DataNotFoundError
from grizz.lazy.ingestor import CsvFileIngestor, CsvIngestor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("frame.csv")
    pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    ).write_csv(path)
    return path


#################################
#     Tests for CsvIngestor     #
#################################


def test_csv_ingestor_repr(frame_path: Path) -> None:
    assert repr(CsvIngestor(frame_path)).startswith("CsvIngestor(")


def test_csv_ingestor_repr_with_kwargs(frame_path: Path) -> None:
    assert repr(CsvIngestor(frame_path, columns=["col1", "col3"])).startswith("CsvIngestor(")


def test_csv_ingestor_str(frame_path: Path) -> None:
    assert str(CsvIngestor(frame_path)).startswith("CsvIngestor(")


def test_csv_ingestor_str_with_kwargs(frame_path: Path) -> None:
    assert str(CsvIngestor(frame_path, columns=["col1", "col3"])).startswith("CsvIngestor(")


def test_csv_ingestor_equal_true(tmp_path: Path) -> None:
    assert CsvIngestor(tmp_path.joinpath("data.csv")).equal(
        CsvIngestor(tmp_path.joinpath("data.csv"))
    )


def test_csv_ingestor_equal_false_different_path(tmp_path: Path) -> None:
    assert not CsvIngestor(tmp_path.joinpath("data.csv")).equal(
        CsvIngestor(tmp_path.joinpath("data2.csv"))
    )


def test_csv_ingestor_equal_false_different_kwargs(tmp_path: Path) -> None:
    assert not CsvIngestor(tmp_path.joinpath("data.csv")).equal(
        CsvIngestor(tmp_path.joinpath("data.csv"), include_header=False)
    )


def test_csv_ingestor_equal_false_different_type(tmp_path: Path) -> None:
    assert not CsvIngestor(tmp_path.joinpath("data.csv")).equal(42)


def test_csv_ingestor_ingest(frame_path: Path) -> None:
    assert_frame_equal(
        CsvIngestor(frame_path).ingest(),
        pl.LazyFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_csv_ingestor_ingest_with_kwargs(frame_path: Path) -> None:
    assert_frame_equal(
        CsvIngestor(frame_path, n_rows=3).ingest(),
        pl.LazyFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
                "col3": [1.2, 2.2, 3.2],
            }
        ),
    )


#####################################
#     Tests for CsvFileIngestor     #
#####################################


def test_csv_file_ingestor_repr(frame_path: Path) -> None:
    assert repr(CsvFileIngestor(frame_path)).startswith("CsvFileIngestor(")


def test_csv_file_ingestor_repr_with_kwargs(frame_path: Path) -> None:
    assert repr(CsvFileIngestor(frame_path, columns=["col1", "col3"])).startswith(
        "CsvFileIngestor("
    )


def test_csv_file_ingestor_str(frame_path: Path) -> None:
    assert str(CsvFileIngestor(frame_path)).startswith("CsvFileIngestor(")


def test_csv_file_ingestor_str_with_kwargs(frame_path: Path) -> None:
    assert str(CsvFileIngestor(frame_path, columns=["col1", "col3"])).startswith("CsvFileIngestor(")


def test_csv_file_ingestor_equal_true(tmp_path: Path) -> None:
    assert CsvFileIngestor(tmp_path.joinpath("data.csv")).equal(
        CsvFileIngestor(tmp_path.joinpath("data.csv"))
    )


def test_csv_file_ingestor_equal_false_different_path(tmp_path: Path) -> None:
    assert not CsvFileIngestor(tmp_path.joinpath("data.csv")).equal(
        CsvFileIngestor(tmp_path.joinpath("data2.csv"))
    )


def test_csv_file_ingestor_equal_false_different_kwargs(tmp_path: Path) -> None:
    assert not CsvFileIngestor(tmp_path.joinpath("data.csv")).equal(
        CsvFileIngestor(tmp_path.joinpath("data.csv"), include_header=False)
    )


def test_csv_file_ingestor_equal_false_different_type(tmp_path: Path) -> None:
    assert not CsvFileIngestor(tmp_path.joinpath("data.csv")).equal(42)


def test_csv_file_ingestor_ingest(frame_path: Path) -> None:
    assert_frame_equal(
        CsvFileIngestor(frame_path).ingest(),
        pl.LazyFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_csv_file_ingestor_ingest_with_kwargs(frame_path: Path) -> None:
    assert_frame_equal(
        CsvFileIngestor(frame_path, n_rows=3).ingest(),
        pl.LazyFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
                "col3": [1.2, 2.2, 3.2],
            }
        ),
    )


def test_csv_file_ingestor_ingest_missing_path(tmp_path: Path) -> None:
    ingestor = CsvFileIngestor(tmp_path.joinpath("data.csv"))
    with pytest.raises(DataNotFoundError, match="Data file does not exist"):
        ingestor.ingest()
