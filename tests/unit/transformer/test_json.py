from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.transformer import JsonDecode


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": ["[]", "[1]", "[1, 2]", "[1, 2, 3]", "[1, 2, 3, 4]"],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "col3": [
                "{'a': 1, 'b': 'abc'}",
                "{'a': 2, 'b': 'def'}",
                "{'a': 0, 'b': ''}",
                "{'a': 1, 'b': 'meow'}",
                "{'a': 0, 'b': ''}",
            ],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.String, "col2": pl.Float32, "col3": pl.String, "col4": pl.String},
    )


###########################################
#     Tests for JsonDecodeTransformer     #
###########################################


def test_json_decode_transformer_repr() -> None:
    assert repr(JsonDecode(columns=["col1", "col3"], prefix="p_", suffix="_s")) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', prefix='p_', suffix='_s', "
        "dtype=None)"
    )


def test_json_decode_transformer_repr_with_kwargs() -> None:
    assert repr(JsonDecode(columns=["col1", "col3"], prefix="p_", suffix="_s", strict=False)) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', prefix='p_', suffix='_s', "
        "dtype=None, strict=False)"
    )


def test_json_decode_transformer_str() -> None:
    assert str(JsonDecode(columns=["col1", "col3"], prefix="p_", suffix="_s")) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', prefix='p_', suffix='_s', "
        "dtype=None)"
    )


def test_json_decode_transformer_str_with_kwargs() -> None:
    assert str(JsonDecode(columns=["col1", "col3"], prefix="p_", suffix="_s", strict=False)) == (
        "JsonDecodeTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', prefix='p_', suffix='_s', "
        "dtype=None, strict=False)"
    )


def test_json_decode_transformer_equal_true() -> None:
    assert JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out")
    )


def test_json_decode_transformer_equal_false_different_columns() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


def test_json_decode_transformer_equal_false_different_prefix() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col3"], prefix="bin_", suffix="_out")
    )


def test_json_decode_transformer_equal_false_different_suffix() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col3"], prefix="", suffix="")
    )


def test_json_decode_transformer_equal_false_different_exclude_columns() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


def test_json_decode_transformer_equal_false_different_exist_policy() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


def test_json_decode_transformer_equal_false_different_missing_policy() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


def test_json_decode_transformer_equal_false_different_kwargs() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out", threshold=1.0)
    )


def test_json_decode_transformer_equal_false_different_type() -> None:
    assert not JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


def test_json_decode_transformer_get_args() -> None:
    assert objects_are_equal(
        JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out", strict=False).get_args(),
        {
            "columns": ("col1", "col3"),
            "prefix": "",
            "suffix": "_out",
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "dtype": None,
            "strict": False,
        },
    )


def test_json_decode_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'JsonDecodeTransformer.fit' as there are no parameters available to fit"
    )


def test_json_decode_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_json_decode_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_json_decode_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_json_decode_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["[]", "[1]", "[1, 2]", "[1, 2, 3]", "[1, 2, 3, 4]"],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    "{'a': 1, 'b': 'abc'}",
                    "{'a': 2, 'b': 'def'}",
                    "{'a': 0, 'b': ''}",
                    "{'a': 1, 'b': 'meow'}",
                    "{'a': 0, 'b': ''}",
                ],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                "col3_out": [
                    {"a": 1, "b": "abc"},
                    {"a": 2, "b": "def"},
                    {"a": 0, "b": ""},
                    {"a": 1, "b": "meow"},
                    {"a": 0, "b": ""},
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.List(pl.Int64),
                "col3_out": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
            },
        ),
    )


def test_json_decode_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["[]", "[1]", "[1, 2]", "[1, 2, 3]", "[1, 2, 3, 4]"],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    "{'a': 1, 'b': 'abc'}",
                    "{'a': 2, 'b': 'def'}",
                    "{'a': 0, 'b': ''}",
                    "{'a': 1, 'b': 'meow'}",
                    "{'a': 0, 'b': ''}",
                ],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                "col3_out": [
                    {"a": 1, "b": "abc"},
                    {"a": 2, "b": "def"},
                    {"a": 0, "b": ""},
                    {"a": 1, "b": "meow"},
                    {"a": 0, "b": ""},
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.List(pl.Int64),
                "col3_out": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
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
    transformer = JsonDecode(columns=["col1"], prefix="", suffix="", exist_policy="ignore")
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
    transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore")
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
    transformer = JsonDecode(
        columns=["col1"], dtype=pl.List(pl.Int32), prefix="", suffix="", exist_policy="ignore"
    )
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


def test_json_decode_transformer_transform_columns_none() -> None:
    transformer = JsonDecode(columns=None, prefix="", suffix="", exist_policy="ignore")
    out = transformer.transform(
        pl.DataFrame(
            {"list": ["[]", "[1]"], "dict": ["{'a': 1, 'b': 'abc'}", "{'a': 2, 'b': 'def'}"]},
            schema={"list": pl.String, "dict": pl.String},
        )
    )
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


def test_json_decode_transformer_transform_exclude_columns() -> None:
    transformer = JsonDecode(
        columns=None,
        exclude_columns=["list", "col1"],
        prefix="",
        suffix="",
        exist_policy="ignore",
    )
    out = transformer.transform(
        pl.DataFrame(
            {"list": ["[]", "[1]"], "dict": ["{'a': 1, 'b': 'abc'}", "{'a': 2, 'b': 'def'}"]},
            schema={"list": pl.String, "dict": pl.String},
        )
    )
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


def test_json_decode_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    {"a": 1, "b": "abc"},
                    {"a": 2, "b": "def"},
                    {"a": 0, "b": ""},
                    {"a": 1, "b": "meow"},
                    {"a": 0, "b": ""},
                ],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.List(pl.Int64),
                "col2": pl.Float32,
                "col3": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
                "col4": pl.String,
            },
        ),
    )


def test_json_decode_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_json_decode_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    {"a": 1, "b": "abc"},
                    {"a": 2, "b": "def"},
                    {"a": 0, "b": ""},
                    {"a": 1, "b": "meow"},
                    {"a": 0, "b": ""},
                ],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.List(pl.Int64),
                "col2": pl.Float32,
                "col3": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
                "col4": pl.String,
            },
        ),
    )


def test_json_decode_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(
        columns=["col1", "col3", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["[]", "[1]", "[1, 2]", "[1, 2, 3]", "[1, 2, 3, 4]"],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    "{'a': 1, 'b': 'abc'}",
                    "{'a': 2, 'b': 'def'}",
                    "{'a': 0, 'b': ''}",
                    "{'a': 1, 'b': 'meow'}",
                    "{'a': 0, 'b': ''}",
                ],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                "col3_out": [
                    {"a": 1, "b": "abc"},
                    {"a": 2, "b": "def"},
                    {"a": 0, "b": ""},
                    {"a": 1, "b": "meow"},
                    {"a": 0, "b": ""},
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.List(pl.Int64),
                "col3_out": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
            },
        ),
    )


def test_json_decode_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_json_decode_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = JsonDecode(
        columns=["col1", "col3", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["[]", "[1]", "[1, 2]", "[1, 2, 3]", "[1, 2, 3, 4]"],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    "{'a': 1, 'b': 'abc'}",
                    "{'a': 2, 'b': 'def'}",
                    "{'a': 0, 'b': ''}",
                    "{'a': 1, 'b': 'meow'}",
                    "{'a': 0, 'b': ''}",
                ],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                "col3_out": [
                    {"a": 1, "b": "abc"},
                    {"a": 2, "b": "def"},
                    {"a": 0, "b": ""},
                    {"a": 1, "b": "meow"},
                    {"a": 0, "b": ""},
                ],
            },
            schema={
                "col1": pl.String,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.String,
                "col1_out": pl.List(pl.Int64),
                "col3_out": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)]),
            },
        ),
    )
