from __future__ import annotations

import logging

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.lazy.transformer import SqlTransformer


@pytest.fixture
def lazyframe() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )


####################################
#     Tests for SqlTransformer     #
####################################


def test_sql_transformer_repr() -> None:
    assert repr(SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")) == (
        "SqlTransformer(\n  (query): SELECT col1, col4 FROM self WHERE col1 > 2\n)"
    )


def test_sql_transformer_str() -> None:
    assert str(SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")) == (
        "SqlTransformer(\n  (query): SELECT col1, col4 FROM self WHERE col1 > 2\n)"
    )


def test_sql_transformer_equal_true() -> None:
    assert SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2").equal(
        SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")
    )


def test_sql_transformer_equal_false_different_query() -> None:
    assert not SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2").equal(
        SqlTransformer(query="")
    )


def test_sql_transformer_equal_false_different_type() -> None:
    assert not SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2").equal(42)


def test_sql_transformer_get_args() -> None:
    assert objects_are_equal(
        SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2").get_args(),
        {"query": "SELECT col1, col4 FROM self WHERE col1 > 2"},
    )


def test_sql_transformer_fit(lazyframe: pl.LazyFrame, caplog: pytest.LogCaptureFixture) -> None:
    transformer = SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")
    with caplog.at_level(logging.INFO):
        transformer.fit(lazyframe)
    assert caplog.messages[0].startswith(
        "Skipping 'SqlTransformer.fit' as there are no parameters available to fit"
    )


def test_sql_transformer_fit_transform(lazyframe: pl.LazyFrame) -> None:
    transformer = SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")
    out = transformer.fit_transform(lazyframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {"col1": [3, 4, 5], "col4": ["c", "d", "e"]},
            schema={"col1": pl.Int64, "col4": pl.String},
        ),
    )


def test_sql_transformer_transform(lazyframe: pl.LazyFrame) -> None:
    transformer = SqlTransformer(query="SELECT col1, col4 FROM self WHERE col1 > 2")
    out = transformer.transform(lazyframe)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {"col1": [3, 4, 5], "col4": ["c", "d", "e"]},
            schema={"col1": pl.Int64, "col4": pl.String},
        ),
    )
