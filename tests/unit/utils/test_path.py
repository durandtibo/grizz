from __future__ import annotations

import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest
from iden.io import save_text

from grizz.utils.path import (
    find_files,
    find_parquet_files,
    human_file_size,
    sanitize_path,
)

#####################################
#     Tests for human_file_size     #
#####################################


def test_human_file_size() -> None:
    assert isinstance(human_file_size(Path(__file__).resolve()), str)


def test_human_file_size_2kb() -> None:
    path = Mock(spec=Path, stat=Mock(return_value=Mock(st_size=2048)))
    sanitize_mock = Mock(return_value=path)
    with patch("grizz.utils.path.sanitize_path", sanitize_mock):
        assert human_file_size(path) == "2.00 KB"
        sanitize_mock.assert_called_once_with(path)


###################################
#     Tests for sanitize_path     #
###################################


def test_sanitize_path_empty_str() -> None:
    assert sanitize_path("") == Path.cwd()


def test_sanitize_path_str() -> None:
    assert sanitize_path("something") == Path.cwd().joinpath("something")


def test_sanitize_path_path(tmp_path: Path) -> None:
    assert sanitize_path(tmp_path) == tmp_path


def test_sanitize_path_resolve() -> None:
    assert sanitize_path(Path("something/./../")) == Path.cwd()


def test_sanitize_path_uri() -> None:
    assert sanitize_path("file:///my/path/something/./../") == Path("/my/path")


################################
#     Tests for find_files     #
################################


@pytest.fixture(scope="module")
def tar_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp")
    save_text("text", path.joinpath("a.txt"))
    path.joinpath("subfolder", "sub").mkdir(parents=True, exist_ok=True)
    with tarfile.open(path.joinpath("data.tar"), "w") as tar:
        tar.add(path.joinpath("a.txt"))
    with tarfile.open(path.joinpath("data2.tar.gz"), "w:gz") as tar:
        tar.add(path.joinpath("a.txt"))
    with tarfile.open(path.joinpath("subfolder", "data.tar"), "w") as tar:
        tar.add(path.joinpath("a.txt"))
    with tarfile.open(path.joinpath("subfolder", "sub", "data.tar"), "w") as tar:
        tar.add(path.joinpath("a.txt"))
    return path


def test_find_files(tar_path: Path) -> None:
    assert sorted(find_files(tar_path, filter_fn=tarfile.is_tarfile)) == [
        tar_path.joinpath("data.tar"),
        tar_path.joinpath("data2.tar.gz"),
        tar_path.joinpath("subfolder", "data.tar"),
        tar_path.joinpath("subfolder", "sub", "data.tar"),
    ]


def test_find_files_recursive_false(tar_path: Path) -> None:
    assert sorted(find_files(tar_path, filter_fn=tarfile.is_tarfile, recursive=False)) == [
        tar_path.joinpath("data.tar"),
        tar_path.joinpath("data2.tar.gz"),
    ]


def test_find_files_file(tar_path: Path) -> None:
    assert find_files(
        tar_path.joinpath("subfolder", "sub", "data.tar"), filter_fn=tarfile.is_tarfile
    ) == [tar_path.joinpath("subfolder", "sub", "data.tar")]


def test_find_files_empty(tmp_path: Path) -> None:
    save_text("text", tmp_path.joinpath("file.txt"))
    assert find_files(tmp_path, filter_fn=tarfile.is_tarfile) == []


########################################
#     Tests for find_parquet_files     #
########################################


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp")
    save_text("text", path.joinpath("a.txt"))
    path.joinpath("subfolder", "sub").mkdir(parents=True, exist_ok=True)
    pl.DataFrame({}).write_parquet(path.joinpath("data.parquet"))
    pl.DataFrame({}).write_parquet(path.joinpath("data2.snappy.parquet"))
    pl.DataFrame({}).write_parquet(path.joinpath("subfolder", "data.parquet"))
    pl.DataFrame({}).write_parquet(path.joinpath("subfolder", "sub", "data.parquet"))
    return path


def test_find_parquet_files(parquet_path: Path) -> None:
    assert sorted(find_parquet_files(parquet_path)) == [
        parquet_path.joinpath("data.parquet"),
        parquet_path.joinpath("data2.snappy.parquet"),
        parquet_path.joinpath("subfolder", "data.parquet"),
        parquet_path.joinpath("subfolder", "sub", "data.parquet"),
    ]


def test_find_parquet_files_recursive_false(parquet_path: Path) -> None:
    assert sorted(find_parquet_files(parquet_path, recursive=False)) == [
        parquet_path.joinpath("data.parquet"),
        parquet_path.joinpath("data2.snappy.parquet"),
    ]


def test_find_parquet_files_file(parquet_path: Path) -> None:
    assert find_parquet_files(parquet_path.joinpath("subfolder", "sub", "data.parquet")) == [
        parquet_path.joinpath("subfolder", "sub", "data.parquet")
    ]


def test_find_parquet_files_empty(tmp_path: Path) -> None:
    save_text("text", tmp_path.joinpath("file.txt"))
    assert find_parquet_files(tmp_path) == []
