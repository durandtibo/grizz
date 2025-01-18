from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import Cast, Sequential


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
        }
    )


###########################################
#     Tests for SequentialTransformer     #
###########################################


def test_sequential_transformer_repr() -> None:
    assert repr(
        Sequential(
            [
                Cast(columns=["col1"], dtype=pl.Float32),
                Cast(columns=["col2"], dtype=pl.Int64),
            ]
        )
    ).startswith("SequentialTransformer(")


def test_sequential_transformer_repr_empty() -> None:
    assert repr(Sequential([])) == "SequentialTransformer()"


def test_sequential_transformer_str() -> None:
    assert str(
        Sequential(
            [
                Cast(columns=["col1"], dtype=pl.Float32),
                Cast(columns=["col2"], dtype=pl.Int64),
            ]
        )
    ).startswith("SequentialTransformer(")


def test_sequential_transformer_str_empty() -> None:
    assert str(Sequential([])) == "SequentialTransformer()"


def test_sequential_transformer_equal_true() -> None:
    assert Sequential(
        [
            Cast(columns=["col1"], dtype=pl.Float32),
            Cast(columns=["col2"], dtype=pl.Int64),
        ]
    ).equal(
        Sequential(
            [
                Cast(columns=["col1"], dtype=pl.Float32),
                Cast(columns=["col2"], dtype=pl.Int64),
            ]
        )
    )


def test_sequential_transformer_equal_false_different_transformers() -> None:
    assert not Sequential(
        [
            Cast(columns=["col1"], dtype=pl.Float32),
            Cast(columns=["col2"], dtype=pl.Int64),
        ]
    ).equal(
        Sequential(
            [
                Cast(columns=["col1"], dtype=pl.Float32),
                Cast(columns=["col2", "col3"], dtype=pl.Int64),
            ]
        )
    )


def test_sequential_transformer_equal_false_different_type() -> None:
    assert not Sequential(
        [
            Cast(columns=["col1"], dtype=pl.Float32),
            Cast(columns=["col2"], dtype=pl.Int64),
        ]
    ).equal(42)


def test_sequential_transformer_fit_1(
    caplog: pytest.LogCaptureFixture, dataframe: pl.DataFrame
) -> None:
    transformer = Sequential([Cast(columns=["col1"], dtype=pl.Float32)])
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'CastTransformer.fit' as there are no parameters available to fit"
    )


def test_sequential_transformer_fit_2(
    caplog: pytest.LogCaptureFixture, dataframe: pl.DataFrame
) -> None:
    transformer = Sequential(
        [
            Cast(columns=["col1"], dtype=pl.Float32),
            Cast(columns=["col2"], dtype=pl.Int64),
        ]
    )
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'CastTransformer.fit' as there are no parameters available to fit"
    )
    assert caplog.messages[1].startswith(
        "Skipping 'CastTransformer.fit' as there are no parameters available to fit"
    )


def test_sequential_transformer_fit_transform_1(dataframe: pl.DataFrame) -> None:
    transformer = Sequential([Cast(columns=["col1"], dtype=pl.Float32)])
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.String},
        ),
    )


def test_sequential_transformer_fit_transform_2(dataframe: pl.DataFrame) -> None:
    transformer = Sequential(
        [
            Cast(columns=["col1"], dtype=pl.Float32),
            Cast(columns=["col2"], dtype=pl.Int64),
        ]
    )
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["a ", " b", "  c  ", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Int64, "col3": pl.String},
        ),
    )


def test_sequential_transformer_transform_1(dataframe: pl.DataFrame) -> None:
    transformer = Sequential([Cast(columns=["col1"], dtype=pl.Float32)])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.String},
        ),
    )


def test_sequential_transformer_transform_2(dataframe: pl.DataFrame) -> None:
    transformer = Sequential(
        [
            Cast(columns=["col1"], dtype=pl.Float32),
            Cast(columns=["col2"], dtype=pl.Int64),
        ]
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["a ", " b", "  c  ", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Int64, "col3": pl.String},
        ),
    )
