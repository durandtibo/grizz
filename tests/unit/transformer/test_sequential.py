from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from grizz.transformer import Cast, Sequential

###########################################
#     Tests for SequentialTransformer     #
###########################################


def test_sequential_dataframe_transformer_repr() -> None:
    assert repr(
        Sequential(
            [
                Cast(columns=["col1"], dtype=pl.Float32),
                Cast(columns=["col2"], dtype=pl.Int64),
            ]
        )
    ).startswith("SequentialTransformer(")


def test_sequential_dataframe_transformer_repr_empty() -> None:
    assert repr(Sequential([])) == "SequentialTransformer()"


def test_sequential_dataframe_transformer_str() -> None:
    assert str(
        Sequential(
            [
                Cast(columns=["col1"], dtype=pl.Float32),
                Cast(columns=["col2"], dtype=pl.Int64),
            ]
        )
    ).startswith("SequentialTransformer(")


def test_sequential_dataframe_transformer_str_empty() -> None:
    assert str(Sequential([])) == "SequentialTransformer()"


def test_sequential_dataframe_transformer_transform_1() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
        }
    )
    transformer = Sequential([Cast(columns=["col1"], dtype=pl.Float32)])
    out = transformer.transform(frame)
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


def test_sequential_dataframe_transformer_transform_2() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
        }
    )
    transformer = Sequential(
        [
            Cast(columns=["col1"], dtype=pl.Float32),
            Cast(columns=["col2"], dtype=pl.Int64),
        ]
    )
    out = transformer.transform(frame)
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
