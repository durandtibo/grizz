from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import JsonDecode


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"list": ["[]", "[1]"], "dict": ["{'a': 1, 'b': 'abc'}", "{'a': 2, 'b': 'def'}"]},
        schema={"list": pl.String, "dict": pl.String},
    )


###########################################
#     Tests for JsonDecodeTransformer     #
###########################################


def test_json_decode_transformer_repr() -> None:
    assert repr(JsonDecode(columns=["col1", "col3"])) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), dtype=None, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_json_decode_transformer_repr_with_kwargs() -> None:
    assert repr(JsonDecode(columns=["col1", "col3"], strict=False)) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), dtype=None, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_json_decode_transformer_str() -> None:
    assert str(JsonDecode(columns=["col1", "col3"])) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), dtype=None, exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_json_decode_transformer_str_with_kwargs() -> None:
    assert str(JsonDecode(columns=["col1", "col3"], strict=False)) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), dtype=None, exclude_columns=(), "
        "missing_policy='raise', strict=False)"
    )


def test_json_decode_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = JsonDecode(columns=["list", "dict"])
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'JsonDecodeTransformer.fit' as there are no parameters available to fit"
    )


def test_json_decode_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict", "missing"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_json_decode_transformer_fit_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict", "missing"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_json_decode_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict", "missing"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_json_decode_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict"])
    out = transformer.fit_transform(dataframe)
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


def test_json_decode_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict"])
    out = transformer.transform(dataframe)
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


def test_json_decode_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode()
    out = transformer.transform(dataframe)
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


def test_json_decode_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(exclude_columns=["list", "col1"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"list": ["[]", "[1]"], "dict": [{"a": 1, "b": "abc"}, {"a": 2, "b": "def"}]},
            schema={
                "list": pl.String,
                "dict": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
            },
        ),
    )


def test_json_decode_transformer_transform_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict", "missing"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
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


def test_json_decode_transformer_transform_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict", "missing"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_json_decode_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["list", "dict", "missing"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
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
