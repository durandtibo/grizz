from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import Function


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
        }
    )


#########################################
#     Tests for FunctionTransformer     #
#########################################


def my_filter(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("col1").is_in({2, 4}))


def test_function_transformer_repr() -> None:
    assert repr(Function(func=my_filter)).startswith("FunctionTransformer(")


def test_function_transformer_str() -> None:
    assert str(Function(func=my_filter)).startswith("FunctionTransformer(")


def test_function_transformer_equal_true() -> None:
    assert Function(func=my_filter).equal(Function(func=my_filter))


def test_function_transformer_equal_false_different_columns() -> None:
    def filter2(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    assert not Function(func=my_filter).equal(Function(func=filter2))


def test_function_transformer_equal_false_different_type() -> None:
    assert not Function(func=my_filter).equal(42)


def test_function_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = Function(func=my_filter)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'FunctionTransformer.fit' as there are no parameters available to fit"
    )


def test_function_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Function(func=my_filter)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(out, pl.DataFrame({"col1": [2, 4], "col2": ["2", "4"]}))


def test_function_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = Function(func=my_filter)
    out = transformer.transform(dataframe)
    assert_frame_equal(out, pl.DataFrame({"col1": [2, 4], "col2": ["2", "4"]}))
