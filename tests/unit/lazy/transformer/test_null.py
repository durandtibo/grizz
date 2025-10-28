from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.lazy.transformer import DropNullRow

############################################
#     Tests for DropNullRowTransformer     #
############################################


@pytest.fixture
def frame_row() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "col1": ["2020-1-1", None, "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, None],
            "col3": [None, None, None, None, None],
        }
    )


def test_drop_null_row_transformer_repr() -> None:
    assert repr(DropNullRow(columns=["col1", "col3"])) == (
        "DropNullRowTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_drop_null_row_transformer_str() -> None:
    assert str(DropNullRow(columns=["col1", "col3"])) == (
        "DropNullRowTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_drop_null_row_transformer_equal_true() -> None:
    assert DropNullRow(columns=["col1", "col3"]).equal(DropNullRow(columns=["col1", "col3"]))


def test_drop_null_row_transformer_equal_false_different_columns() -> None:
    assert not DropNullRow(columns=["col1", "col3"]).equal(
        DropNullRow(columns=["col1", "col2", "col3"])
    )


def test_drop_null_row_transformer_equal_false_different_exclude_columns() -> None:
    assert not DropNullRow(columns=["col1", "col3"]).equal(
        DropNullRow(columns=["col1", "col3"], exclude_columns=["col2"])
    )


def test_drop_null_row_transformer_equal_false_different_missing_policy() -> None:
    assert not DropNullRow(columns=["col1", "col3"]).equal(
        DropNullRow(columns=["col1", "col3"], missing_policy="warn")
    )


def test_drop_null_row_transformer_equal_false_different_type() -> None:
    assert not DropNullRow(columns=["col1", "col3"]).equal(42)


def test_drop_null_row_transformer_fit(
    frame_row: pl.LazyFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DropNullRow()
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_row)
    assert caplog.messages[0].startswith(
        "Skipping 'DropNullRowTransformer.fit' as there are no parameters available to fit"
    )


def test_drop_null_row_transformer_fit_missing_policy_ignore(
    frame_row: pl.LazyFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(frame_row)


def test_drop_null_row_transformer_fit_missing_policy_raise(
    frame_row: pl.LazyFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(frame_row)


def test_drop_null_row_transformer_fit_missing_policy_warn(
    frame_row: pl.LazyFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(frame_row)


def test_drop_null_row_transformer_fit_transform(frame_row: pl.LazyFrame) -> None:
    transformer = DropNullRow()
    out = transformer.fit_transform(frame_row)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform(frame_row: pl.LazyFrame) -> None:
    transformer = DropNullRow()
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_columns(frame_row: pl.LazyFrame) -> None:
    transformer = DropNullRow(columns=["col2", "col3"])
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": ["2020-1-1", "2020-1-31"],
                "col2": [1, 3],
                "col3": [None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_exclude_columns(frame_row: pl.LazyFrame) -> None:
    transformer = DropNullRow(exclude_columns=["col2"])
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_empty_row() -> None:
    transformer = DropNullRow()
    out = transformer.transform(pl.LazyFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.LazyFrame({"col1": [], "col2": [], "col3": []}))


def test_drop_null_row_transformer_transform_empty() -> None:
    transformer = DropNullRow()
    out = transformer.transform(pl.LazyFrame({}))
    assert_frame_equal(out, pl.LazyFrame({}))


def test_drop_null_row_transformer_transform_missing_policy_ignore(
    frame_row: pl.LazyFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_missing_policy_raise(
    frame_row: pl.LazyFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(frame_row)


def test_drop_null_row_transformer_transform_missing_policy_warn(
    frame_row: pl.LazyFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.LazyFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )
