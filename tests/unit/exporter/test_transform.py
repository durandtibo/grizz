from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exporter import ParquetExporter, TransformExporter
from grizz.transformer import Cast

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )


#######################################
#     Tests for TransformExporter     #
#######################################


def test_transform_exporter_repr(tmp_path: Path) -> None:
    assert repr(
        TransformExporter(
            transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
            exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
        )
    ).startswith("TransformExporter(")


def test_transform_exporter_str(tmp_path: Path) -> None:
    assert str(
        TransformExporter(
            exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
            transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
        )
    ).startswith("TransformExporter(")


def test_transform_exporter_equal_true(tmp_path: Path) -> None:
    assert TransformExporter(
        exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
        transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
    ).equal(
        TransformExporter(
            exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
            transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
        )
    )


def test_transform_exporter_equal_false_different_exporter(tmp_path: Path) -> None:
    assert not TransformExporter(
        exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
        transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
    ).equal(
        TransformExporter(
            exporter=ParquetExporter(path=tmp_path.joinpath("data2.parquet")),
            transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
        )
    )


def test_transform_exporter_equal_false_different_transformer(tmp_path: Path) -> None:
    assert not TransformExporter(
        exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
        transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
    ).equal(
        TransformExporter(
            exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
            transformer=Cast(columns=["col1", "col3"], dtype=pl.Int32),
        )
    )


def test_transform_exporter_equal_false_different_type(tmp_path: Path) -> None:
    assert not TransformExporter(
        exporter=ParquetExporter(path=tmp_path.joinpath("data.parquet")),
        transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
    ).equal(42)


def test_transform_exporter_ingest(tmp_path: Path, dataframe: pl.DataFrame) -> None:
    path = tmp_path.joinpath("data.parquet")
    assert not path.is_file()
    TransformExporter(
        transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
        exporter=ParquetExporter(path=path),
    ).export(dataframe)
    assert path.is_file()
    assert_frame_equal(
        pl.read_parquet(path),
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.Float32},
        ),
    )
