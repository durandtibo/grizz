from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import StripChars


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )


####################################################
#     Tests for StripCharsDataFrameTransformer     #
####################################################


def test_strip_chars_dataframe_transformer_repr() -> None:
    assert repr(StripChars(columns=["col1", "col3"])) == (
        "StripCharsDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False)"
    )


def test_strip_chars_dataframe_transformer_repr_with_kwargs() -> None:
    assert repr(StripChars(columns=["col1", "col3"], characters=None)) == (
        "StripCharsDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False, "
        "characters=None)"
    )


def test_strip_chars_dataframe_transformer_str() -> None:
    assert str(StripChars(columns=["col1", "col3"])) == (
        "StripCharsDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False)"
    )


def test_strip_chars_dataframe_transformer_str_with_kwargs() -> None:
    assert str(StripChars(columns=["col1", "col3"], characters=None)) == (
        "StripCharsDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False, "
        "characters=None)"
    )


def test_strip_chars_dataframe_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
            }
        ),
    )


def test_strip_chars_dataframe_transformer_transform_none() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None],
            "col2": ["1", "2", "3", "4", "5", None],
            "col3": ["a ", " b", "  c  ", "d", "e", None],
            "col4": ["a ", " b", "  c  ", "d", "e", None],
        }
    )
    transformer = StripChars(columns=["col2", "col3"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": ["1", "2", "3", "4", "5", None],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": ["a ", " b", "  c  ", "d", "e", None],
            }
        ),
    )


def test_strip_chars_dataframe_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = StripChars()
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_strip_chars_dataframe_transformer_transform_empty() -> None:
    transformer = StripChars(columns=[])
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_strip_chars_dataframe_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.String})
    transformer = StripChars(columns=["col"])
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": []}, schema={"col": pl.String}))


def test_strip_chars_dataframe_transformer_transform_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StripChars(columns=["col2", "col3", "col5"])
    with pytest.raises(RuntimeError, match="column col5 is not in the DataFrame"):
        transformer.transform(dataframe)


def test_strip_chars_dataframe_transformer_transform_ignore_missing_true(
    caplog: pytest.LogCaptureFixture, dataframe: pl.DataFrame
) -> None:
    transformer = StripChars(columns=["col2", "col3", "col5"], ignore_missing=True)
    with caplog.at_level(logging.WARNING):
        out = transformer.transform(dataframe)
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": ["a ", " b", "  c  ", "d", "e"],
                }
            ),
        )
        assert caplog.messages[-1].startswith(
            "skipping transformation for column col5 because the column is missing"
        )
