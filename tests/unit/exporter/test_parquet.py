from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exporter import ParquetExporter

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
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


def test_parquet_exporter_export(tmp_path: Path, dataframe: pl.DataFrame) -> None:
    path = tmp_path.joinpath("data.parquet")
    assert not path.is_file()
    ParquetExporter(path).export(dataframe)
    assert path.is_file()


def test_parquet_exporter_export_with_kwargs(tmp_path: Path, dataframe: pl.DataFrame) -> None:
    path = tmp_path.joinpath("data.parquet")
    assert not path.is_file()
    ParquetExporter(path, compression="gzip").export(dataframe)
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
