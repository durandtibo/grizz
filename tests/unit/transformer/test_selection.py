from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import ColumnSelection


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )


################################################
#     Tests for ColumnSelectionTransformer     #
################################################


def test_column_selection_transformer_repr() -> None:
    assert (
        repr(ColumnSelection(columns=["col1", "col2"]))
        == "ColumnSelectionTransformer(columns=('col1', 'col2'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_column_selection_transformer_str() -> None:
    assert (
        str(ColumnSelection(columns=["col1", "col2"]))
        == "ColumnSelectionTransformer(columns=('col1', 'col2'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_column_selection_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnSelectionTransformer.fit' as there are no parameters available to fit"
    )


def test_column_selection_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnSelection(columns=["col", "col1"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_selection_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnSelection(columns=["col", "col1"])
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_column_selection_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnSelection(columns=["col", "col1"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_column_selection_fit_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_column_selection_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_column_selection_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = ColumnSelection()
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
                "col3": [None, None, None, None, None],
            }
        ),
    )


def test_column_selection_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = ColumnSelection(exclude_columns=["col3", "col4"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_column_selection_transformer_transform_empty_columns(dataframe: pl.DataFrame) -> None:
    transformer = ColumnSelection(columns=[])
    out = transformer.transform(dataframe)
    assert_frame_equal(out, pl.DataFrame({}))


def test_column_selection_transformer_transform_empty_row() -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": []}))


def test_column_selection_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnSelection(columns=["col", "col1"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame({"col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None]}),
    )


def test_column_selection_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnSelection(columns=["col", "col1"])
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_column_selection_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnSelection(columns=["col", "col1"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame({"col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None]}),
    )
