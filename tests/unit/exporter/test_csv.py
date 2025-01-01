from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from grizz.exporter import CsvExporter

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


def test_csv_exporter_export(tmp_path: Path, dataframe: pl.DataFrame) -> None:
    path = tmp_path.joinpath("my_folder/data.csv")
    assert not path.is_file()
    CsvExporter(path).export(dataframe)
    assert path.is_file()


def test_csv_exporter_export_with_kwargs(tmp_path: Path, dataframe: pl.DataFrame) -> None:
    path = tmp_path.joinpath("my_folder/data.csv")
    assert not path.is_file()
    CsvExporter(path, include_header=False).export(dataframe)
    assert path.is_file()
