from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import DataFrameNotFoundError
from grizz.exporter import InMemoryExporter


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


######################################
#     Tests for InMemoryExporter     #
######################################


def test_parquet_exporter_repr() -> None:
    assert repr(InMemoryExporter()) == "InMemoryExporter(frame=None)"


def test_parquet_exporter_repr_with_frame(dataframe: pl.DataFrame) -> None:
    exporter = InMemoryExporter()
    exporter.export(dataframe)
    assert repr(exporter) == "InMemoryExporter(frame=(5, 3))"


def test_parquet_exporter_str() -> None:
    assert str(InMemoryExporter()) == "InMemoryExporter(frame=None)"


def test_parquet_exporter_str_with_frame(dataframe: pl.DataFrame) -> None:
    exporter = InMemoryExporter()
    exporter.export(dataframe)
    assert str(exporter) == "InMemoryExporter(frame=(5, 3))"


def test_parquet_exporter_export(dataframe: pl.DataFrame) -> None:
    exporter = InMemoryExporter()
    exporter.export(dataframe)
    assert_frame_equal(
        exporter._frame,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
        ),
    )


def test_parquet_exporter_export_overwrite(dataframe: pl.DataFrame) -> None:
    exporter = InMemoryExporter()
    exporter.export(pl.DataFrame({}))
    assert_frame_equal(exporter._frame, pl.DataFrame({}))
    exporter.export(dataframe)
    assert_frame_equal(
        exporter._frame,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
        ),
    )


def test_parquet_exporter_ingest(dataframe: pl.DataFrame) -> None:
    exporter = InMemoryExporter()
    exporter.export(dataframe)
    assert_frame_equal(
        exporter.ingest(),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
        ),
    )


def test_parquet_exporter_ingest_empty() -> None:
    exporter = InMemoryExporter()
    with pytest.raises(DataFrameNotFoundError, match="No DataFrame available for ingestion."):
        exporter.ingest()
