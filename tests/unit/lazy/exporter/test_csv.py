from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.lazy.exporter import CsvExporter

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


#################################
#     Tests for CsvExporter     #
#################################


def test_csv_exporter_repr(tmp_path: Path) -> None:
    assert repr(CsvExporter(tmp_path.joinpath("data.csv"))).startswith("CsvExporter(")


def test_csv_exporter_repr_with_kwargs(tmp_path: Path) -> None:
    assert repr(CsvExporter(tmp_path.joinpath("data.csv"), include_header=False)).startswith(
        "CsvExporter("
    )


def test_csv_exporter_str(tmp_path: Path) -> None:
    assert str(CsvExporter(tmp_path.joinpath("data.csv"))).startswith("CsvExporter(")


def test_csv_exporter_str_with_kwargs(tmp_path: Path) -> None:
    assert str(CsvExporter(tmp_path.joinpath("data.csv"), include_header=False)).startswith(
        "CsvExporter("
    )


def test_csv_exporter_equal_true(tmp_path: Path) -> None:
    assert CsvExporter(tmp_path.joinpath("data.csv")).equal(
        CsvExporter(tmp_path.joinpath("data.csv"))
    )


def test_csv_exporter_equal_false_different_path(tmp_path: Path) -> None:
    assert not CsvExporter(tmp_path.joinpath("data.csv")).equal(
        CsvExporter(tmp_path.joinpath("data2.csv"))
    )


def test_csv_exporter_equal_false_different_kwargs(tmp_path: Path) -> None:
    assert not CsvExporter(tmp_path.joinpath("data.csv")).equal(
        CsvExporter(tmp_path.joinpath("data.csv"), include_header=False)
    )


def test_csv_exporter_equal_false_different_type(tmp_path: Path) -> None:
    assert not CsvExporter(tmp_path.joinpath("data.csv")).equal(42)


def test_csv_exporter_export(tmp_path: Path, lazyframe: pl.LazyFrame) -> None:
    path = tmp_path.joinpath("my_folder/data.csv")
    assert not path.is_file()
    CsvExporter(path).export(lazyframe)
    assert path.is_file()

    assert_frame_equal(
        pl.read_csv(path),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
        ),
    )


def test_csv_exporter_export_with_kwargs(tmp_path: Path, lazyframe: pl.LazyFrame) -> None:
    separator = ","
    path = tmp_path.joinpath("my_folder/data.csv")
    assert not path.is_file()
    CsvExporter(path, separator=separator).export(lazyframe)
    assert path.is_file()

    assert_frame_equal(
        pl.read_csv(path, separator=separator),
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.Float64},
        ),
    )
