from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import DropDuplicate

##############################################
#     Tests for DropDuplicateTransformer     #
##############################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 1],
            "col2": ["1", "2", "3", "4", "1"],
            "col3": ["1", "2", "3", "1", "1"],
            "col4": ["a", "a", "a", "a", "a"],
        }
    )


def test_drop_duplicate_transformer_repr() -> None:
    assert repr(DropDuplicate()) == "DropDuplicateTransformer(columns=None, ignore_missing=False)"


def test_drop_duplicate_transformer_repr_with_kwargs() -> None:
    assert repr(DropDuplicate(columns=["col1", "col3"], keep="first")) == (
        "DropDuplicateTransformer(columns=('col1', 'col3'), ignore_missing=False, keep=first)"
    )


def test_drop_duplicate_transformer_str() -> None:
    assert str(DropDuplicate()) == "DropDuplicateTransformer(columns=None, ignore_missing=False)"


def test_drop_duplicate_transformer_str_with_kwargs() -> None:
    assert str(DropDuplicate(columns=["col1", "col3"], keep="first")) == (
        "DropDuplicateTransformer(columns=('col1', 'col3'), ignore_missing=False, keep=first)"
    )


def test_drop_duplicate_transformer_transform_threshold_1(dataframe: pl.DataFrame) -> None:
    transformer = DropDuplicate(keep="first", maintain_order=True)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4],
                "col2": ["1", "2", "3", "4"],
                "col3": ["1", "2", "3", "1"],
                "col4": ["a", "a", "a", "a"],
            }
        ),
    )


def test_drop_duplicate_transformer_transform_columns(dataframe: pl.DataFrame) -> None:
    transformer = DropDuplicate(columns=["col3", "col4"], keep="first", maintain_order=True)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [
                    "1",
                    "2",
                    "3",
                ],
                "col3": ["1", "2", "3"],
                "col4": ["a", "a", "a"],
            }
        ),
    )


def test_drop_duplicate_transformer_transform_empty_row() -> None:
    transformer = DropDuplicate()
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_drop_duplicate_transformer_transform_empty() -> None:
    transformer = DropDuplicate()
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_drop_duplicate_transformer_transform_columns_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropDuplicate(columns=["col1", "col2", "col5"])
    with pytest.raises(RuntimeError, match="1 columns are missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_drop_duplicate_transformer_transform_columns_ignore_missing_true(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DropDuplicate(
        columns=["col1", "col2", "col5"], ignore_missing=True, keep="first", maintain_order=True
    )
    with caplog.at_level(logging.WARNING):
        out = transformer.transform(dataframe)
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4],
                    "col2": ["1", "2", "3", "4"],
                    "col3": ["1", "2", "3", "1"],
                    "col4": ["a", "a", "a", "a"],
                }
            ),
        )
        assert caplog.messages[-1].startswith(
            "1 columns are missing in the DataFrame and will be ignored:"
        )
