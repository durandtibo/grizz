from __future__ import annotations

import polars as pl
import pytest

from grizz.transformer import BaseInNTransformer


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )


########################################
#     Tests for BaseInNTransformer     #
########################################


class MyColumnsTransformer(BaseInNTransformer):

    def _fit(self, frame: pl.DataFrame) -> None:
        pass

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame


def test_base_columns_transformer_repr() -> None:
    assert (
        repr(MyColumnsTransformer())
        == "MyColumnsTransformer(columns=None, exclude_columns=(), missing_policy='raise')"
    )


def test_base_columns_transformer_str() -> None:
    assert (
        str(MyColumnsTransformer())
        == "MyColumnsTransformer(columns=None, exclude_columns=(), missing_policy='raise')"
    )


def test_base_columns_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer(columns=["col2", "col3", "col5"])
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_base_columns_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer()
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_base_columns_transformer_find_columns_exclude(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer(
        columns=["col2", "col3", "col5"], exclude_columns=["col3", "col6"]
    )
    assert transformer.find_columns(dataframe) == ("col2", "col5")


def test_base_columns_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer(columns=["col2", "col3", "col5"])
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_base_columns_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer()
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_base_columns_transformer_find_common_columns_exclude(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer(exclude_columns=["col3", "col6"])
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col4")


def test_base_columns_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer(columns=["col2", "col3", "col5"])
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_base_columns_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = MyColumnsTransformer()
    assert transformer.find_missing_columns(dataframe) == ()
