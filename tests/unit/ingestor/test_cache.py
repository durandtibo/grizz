from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import DataFrameNotFoundError
from grizz.exporter import ParquetExporter
from grizz.ingestor import CacheIngestor, CsvIngestor, ParquetIngestor

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


###################################
#     Tests for CacheIngestor     #
###################################


def test_cache_ingestor_repr(tmp_path: Path, frame_path: Path) -> None:
    path = tmp_path.joinpath("data.parquet")
    ingestor = CacheIngestor(
        ingestor_slow=CsvIngestor(frame_path),
        ingestor_fast=ParquetIngestor(path),
        exporter=ParquetExporter(path),
    )
    assert repr(ingestor).startswith("CacheIngestor(")


def test_cache_ingestor_str(tmp_path: Path, frame_path: Path) -> None:
    path = tmp_path.joinpath("data.parquet")
    ingestor = CacheIngestor(
        ingestor_slow=CsvIngestor(frame_path),
        ingestor_fast=ParquetIngestor(path),
        exporter=ParquetExporter(path),
    )
    assert str(ingestor).startswith("CacheIngestor(")


def test_cache_ingestor_equal_true(tmp_path: Path) -> None:
    assert CacheIngestor(
        ingestor_slow=CsvIngestor(tmp_path.joinpath("data.csv")),
        ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
        exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
    ).equal(
        CacheIngestor(
            ingestor_slow=CsvIngestor(tmp_path.joinpath("data.csv")),
            ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
            exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
        )
    )


def test_cache_ingestor_equal_false_different_ingestor_slow(tmp_path: Path) -> None:
    assert not CacheIngestor(
        ingestor_slow=CsvIngestor(tmp_path.joinpath("data.csv")),
        ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
        exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
    ).equal(
        CacheIngestor(
            ingestor_slow=CsvIngestor(tmp_path.joinpath("data2.csv")),
            ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
            exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
        )
    )


def test_cache_ingestor_equal_false_different_ingestor_fast(tmp_path: Path) -> None:
    assert not CacheIngestor(
        ingestor_slow=CsvIngestor(tmp_path.joinpath("data.csv")),
        ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
        exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
    ).equal(
        CacheIngestor(
            ingestor_slow=CsvIngestor(tmp_path.joinpath("data.csv")),
            ingestor_fast=ParquetIngestor(tmp_path.joinpath("data2.parquet")),
            exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
        )
    )


def test_cache_ingestor_equal_false_different_exporter(tmp_path: Path) -> None:
    assert not CacheIngestor(
        ingestor_slow=CsvIngestor(tmp_path.joinpath("data.csv")),
        ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
        exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
    ).equal(
        CacheIngestor(
            ingestor_slow=CsvIngestor(tmp_path.joinpath("data2.csv")),
            ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
            exporter=ParquetExporter(tmp_path.joinpath("data2.parquet")),
        )
    )


def test_cache_ingestor_equal_false_different_type(tmp_path: Path) -> None:
    assert not CacheIngestor(
        ingestor_slow=CsvIngestor(tmp_path.joinpath("data.csv")),
        ingestor_fast=ParquetIngestor(tmp_path.joinpath("data.parquet")),
        exporter=ParquetExporter(tmp_path.joinpath("data.parquet")),
    ).equal(42)


def test_cache_ingestor_ingest_slow(tmp_path: Path, frame_path: Path) -> None:
    path = tmp_path.joinpath("data.parquet")
    ingestor = CacheIngestor(
        ingestor_slow=CsvIngestor(frame_path),
        ingestor_fast=ParquetIngestor(path),
        exporter=ParquetExporter(path),
    )
    assert not path.is_file()
    out = ingestor.ingest()
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
    assert_frame_equal(
        pl.read_parquet(path),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_cache_ingestor_ingest_fast(tmp_path: Path, frame_path: Path) -> None:
    path = tmp_path.joinpath("data.parquet")
    pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        },
        schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.Float32},
    ).write_parquet(path)
    ingestor = CacheIngestor(
        ingestor_slow=CsvIngestor(frame_path),
        ingestor_fast=ParquetIngestor(path),
        exporter=ParquetExporter(path),
    )
    assert path.is_file()
    out = ingestor.ingest()
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.Float32},
        ),
    )


def test_cache_ingestor_ingest_missing_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data")
    ingestor = CacheIngestor(
        ingestor_slow=CsvIngestor(path),
        ingestor_fast=ParquetIngestor(path),
        exporter=ParquetExporter(path),
    )
    with pytest.raises(DataFrameNotFoundError, match="DataFrame file does not exist"):
        ingestor.ingest()
