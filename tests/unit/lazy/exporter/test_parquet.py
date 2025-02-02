from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.lazy.exporter import ParquetExporter

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def lazyframe() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
    )


#####################################
#     Tests for ParquetExporter     #
#####################################


def test_parquet_exporter_repr(tmp_path: Path) -> None:
    assert repr(ParquetExporter(tmp_path.joinpath("data.parquet"))).startswith("ParquetExporter(")


def test_parquet_exporter_repr_with_kwargs(tmp_path: Path) -> None:
    assert repr(ParquetExporter(tmp_path.joinpath("data.parquet"), compression="gzip")).startswith(
        "ParquetExporter("
    )


def test_parquet_exporter_str(tmp_path: Path) -> None:
    assert str(ParquetExporter(tmp_path.joinpath("data.parquet"))).startswith("ParquetExporter(")


def test_parquet_exporter_str_with_kwargs(tmp_path: Path) -> None:
    assert str(ParquetExporter(tmp_path.joinpath("data.parquet"), compression="gzip")).startswith(
        "ParquetExporter("
    )


def test_parquet_exporter_equal_true(tmp_path: Path) -> None:
    assert ParquetExporter(tmp_path.joinpath("data.parquet")).equal(
        ParquetExporter(tmp_path.joinpath("data.parquet"))
    )


def test_parquet_exporter_equal_false_different_path(tmp_path: Path) -> None:
    assert not ParquetExporter(tmp_path.joinpath("data.parquet")).equal(
        ParquetExporter(tmp_path.joinpath("data2.parquet"))
    )


def test_parquet_exporter_equal_false_different_kwargs(tmp_path: Path) -> None:
    assert not ParquetExporter(tmp_path.joinpath("data.parquet")).equal(
        ParquetExporter(tmp_path.joinpath("data.parquet"), include_header=False)
    )


def test_parquet_exporter_equal_false_different_type(tmp_path: Path) -> None:
    assert not ParquetExporter(tmp_path.joinpath("data.parquet")).equal(42)


def test_parquet_exporter_export(tmp_path: Path, lazyframe: pl.LazyFrame) -> None:
    path = tmp_path.joinpath("my_folder/data.parquet")
    assert not path.is_file()
    ParquetExporter(path).export(lazyframe)
    assert path.is_file()

    assert_frame_equal(
        pl.read_parquet(path),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
        ),
    )


def test_parquet_exporter_export_with_kwargs(tmp_path: Path, lazyframe: pl.LazyFrame) -> None:
    path = tmp_path.joinpath("my_folder/data.parquet")
    assert not path.is_file()
    ParquetExporter(path, compression="gzip").export(lazyframe)
    assert path.is_file()

    assert_frame_equal(
        pl.read_parquet(path),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
        ),
    )
