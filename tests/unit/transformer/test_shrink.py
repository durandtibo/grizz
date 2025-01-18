from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import ShrinkMemory


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )


#############################################
#     Tests for ShrinkMemoryTransformer     #
#############################################


def test_shrink_memory_transformer_repr() -> None:
    assert repr(ShrinkMemory()) == ("ShrinkMemoryTransformer()")


def test_shrink_memory_transformer_str() -> None:
    assert str(ShrinkMemory()) == ("ShrinkMemoryTransformer()")


def test_shrink_memory_transformer_equal_true() -> None:
    assert ShrinkMemory().equal(ShrinkMemory())


def test_shrink_memory_transformer_equal_false_different_type() -> None:
    assert not ShrinkMemory().equal(42)


def test_shrink_memory_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ShrinkMemory()
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ShrinkMemoryTransformer.fit' as there are no parameters available to fit"
    )


def test_shrink_memory_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ShrinkMemory()
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_shrink_memory_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ShrinkMemory()
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )
