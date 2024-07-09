from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import NullColumn

###########################################
#     Tests for NullColumnTransformer     #
###########################################


@pytest.fixture()
def frame_col() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )


def test_null_column_transformer_repr() -> None:
    assert repr(NullColumn(columns=["col1", "col3"])) == (
        "NullColumnTransformer(columns=('col1', 'col3'), threshold=1.0, ignore_missing=False)"
    )


def test_null_column_transformer_repr_with_kwargs() -> None:
    assert repr(NullColumn(columns=["col1", "col3"], strict=False)) == (
        "NullColumnTransformer(columns=('col1', 'col3'), threshold=1.0, ignore_missing=False, "
        "strict=False)"
    )


def test_null_column_transformer_str() -> None:
    assert str(NullColumn(columns=["col1", "col3"])) == (
        "NullColumnTransformer(columns=('col1', 'col3'), threshold=1.0, ignore_missing=False)"
    )


def test_null_column_transformer_str_with_kwargs() -> None:
    assert str(NullColumn(columns=["col1", "col3"], strict=False)) == (
        "NullColumnTransformer(columns=('col1', 'col3'), threshold=1.0, ignore_missing=False, "
        "strict=False)"
    )


def test_null_column_transformer_transform_threshold_1(frame_col: pl.DataFrame) -> None:
    transformer = NullColumn(threshold=1.0)
    out = transformer.transform(frame_col)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_null_column_transformer_transform_threshold_0_4(frame_col: pl.DataFrame) -> None:
    transformer = NullColumn(threshold=0.4)
    out = transformer.transform(frame_col)
    assert_frame_equal(
        out, pl.DataFrame({"col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None]})
    )


def test_null_column_transformer_transform_threshold_0_2(frame_col: pl.DataFrame) -> None:
    transformer = NullColumn(threshold=0.2)
    out = transformer.transform(frame_col)
    assert_frame_equal(out, pl.DataFrame({}))


def test_null_column_transformer_transform_columns(frame_col: pl.DataFrame) -> None:
    transformer = NullColumn(columns=["col1", "col2"], threshold=0.4)
    out = transformer.transform(frame_col)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col3": [None, None, None, None, None],
            }
        ),
    )


def test_null_column_transformer_transform_empty_row() -> None:
    transformer = NullColumn(threshold=0.5)
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_null_column_transformer_transform_empty() -> None:
    transformer = NullColumn(threshold=0.5)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_null_column_transformer_transform_columns_ignore_missing_false(
    frame_col: pl.DataFrame,
) -> None:
    transformer = NullColumn(columns=["col1", "col2", "col5"], threshold=0.4)
    with pytest.raises(RuntimeError, match="1 columns are missing in the DataFrame:"):
        transformer.transform(frame_col)


def test_null_column_transformer_transform_columns_ignore_missing_true(
    frame_col: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = NullColumn(columns=["col1", "col2", "col5"], threshold=0.4, ignore_missing=True)
    with caplog.at_level(logging.WARNING):
        out = transformer.transform(frame_col)
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                    "col3": [None, None, None, None, None],
                }
            ),
        )
        assert caplog.messages[-1].startswith(
            "1 columns are missing in the DataFrame and will be ignored:"
        )
