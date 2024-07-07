from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from grizz.transformer import JsonDecode

###########################################
#     Tests for JsonDecodeTransformer     #
###########################################


def test_json_decode_transformer_str() -> None:
    assert str(JsonDecode(columns=["col1", "col3"])).startswith("JsonDecodeTransformer(")


def test_json_decode_transformer_transform() -> None:
    frame = pl.DataFrame(
        {"list": ["[]", "[1]"], "dict": ["{'a': 1, 'b': 'abc'}", "{'a': 2, 'b': 'def'}"]},
        schema={"list": pl.String, "dict": pl.String},
    )
    transformer = JsonDecode(columns=["list", "dict"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"list": [[], [1]], "dict": [{"a": 1, "b": "abc"}, {"a": 2, "b": "def"}]},
            schema={
                "list": pl.List(pl.Int64),
                "dict": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
            },
        ),
    )


def test_json_decode_transformer_transform_one_col() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["[1, 2]", "[2]", "[1, 2, 3]", "[4, 5]", "[5, 4]"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["['1', '2']", "['2']", "['1', '2', '3']", "['4', '5']", "['5', '4']"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = JsonDecode(columns=["col1"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [[1, 2], [2], [1, 2, 3], [4, 5], [5, 4]],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["['1', '2']", "['2']", "['1', '2', '3']", "['4', '5']", "['5', '4']"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.List(pl.Int64),
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
            },
        ),
    )


def test_json_decode_transformer_transform_two_cols() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["[1, 2]", "[2]", "[1, 2, 3]", "[4, 5]", "[5, 4]"],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [r"['1', '2']", r"['2']", r"['1', '2', '3']", r"['4', '5']", r"['5', '4']"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )
    transformer = JsonDecode(columns=["col1", "col3"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [[1, 2], [2], [1, 2, 3], [4, 5], [5, 4]],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [["1", "2"], ["2"], ["1", "2", "3"], ["4", "5"], ["5", "4"]],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.List(pl.Int64),
                "col2": pl.String,
                "col3": pl.List(pl.String),
                "col4": pl.String,
            },
        ),
    )


def test_json_decode_transformer_transform_dtype() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["[1, 2]", "[2]", "[1, 2, 3]", "[4, 5]", "[5, 4]"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    transformer = JsonDecode(columns=["col1"], dtype=pl.List(pl.Int32))
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [[1, 2], [2], [1, 2, 3], [4, 5], [5, 4]],
                "col2": ["1", "2", "3", "4", "5"],
            },
            schema={
                "col1": pl.List(pl.Int32),
                "col2": pl.String,
            },
        ),
    )
