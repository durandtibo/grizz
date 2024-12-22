from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import ColumnSelection

################################################
#     Tests for ColumnSelectionTransformer     #
################################################


def test_column_selection_transformer_repr() -> None:
    assert repr(ColumnSelection(columns=["col1", "col2"])).startswith("ColumnSelectionTransformer(")


def test_column_selection_transformer_str() -> None:
    assert str(ColumnSelection(columns=["col1", "col2"])).startswith("ColumnSelectionTransformer(")


def test_column_selection_transformer_transform() -> None:
    frame = pl.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    transformer = ColumnSelection(columns=["col1", "col2"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_column_selection_transformer_transform_empty_row() -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": []}))


def test_column_selection_transformer_transform_empty() -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    with pytest.raises(ColumnNotFoundError, match=r"2 columns are missing in the DataFrame:"):
        transformer.transform(pl.DataFrame({}))


def test_column_selection_transformer_transform_ignore_missing_true() -> None:
    transformer = ColumnSelection(columns=["col"], ignore_missing=True)
    with pytest.warns(
        ColumnNotFoundWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))
