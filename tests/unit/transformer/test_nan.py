from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import DropNanRow

###########################################
#     Tests for DropNanRowTransformer     #
###########################################


@pytest.fixture
def frame_row() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
            "col2": [1.0, float("nan"), 3.0, float("nan"), float("nan")],
            "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
        }
    )


def test_drop_nan_row_transformer_repr() -> None:
    assert repr(DropNanRow(columns=["col1", "col3"])) == (
        "DropNanRowTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_drop_nan_row_transformer_str() -> None:
    assert str(DropNanRow(columns=["col1", "col3"])) == (
        "DropNanRowTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_drop_nan_row_transformer_fit(
    frame_row: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DropNanRow()
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_row)
    assert caplog.messages[0].startswith(
        "Skipping 'DropNanRowTransformer.fit' as there are no parameters available to fit"
    )


def test_drop_nan_row_transformer_fit_missing_policy_ignore(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(frame_row)


def test_drop_nan_row_transformer_fit_missing_policy_raise(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(frame_row)


def test_drop_nan_row_transformer_fit_missing_policy_warn(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(frame_row)


def test_drop_nan_row_transformer_fit_transform(frame_row: pl.DataFrame) -> None:
    transformer = DropNanRow()
    out = transformer.fit_transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": [1.0, float("nan"), 3.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan")],
            }
        ),
    )


def test_drop_nan_row_transformer_transform(frame_row: pl.DataFrame) -> None:
    transformer = DropNanRow()
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": [1.0, float("nan"), 3.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan")],
            }
        ),
    )


def test_drop_nan_row_transformer_transform_columns(frame_row: pl.DataFrame) -> None:
    transformer = DropNanRow(columns=["col2", "col3"])
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 3.0],
                "col2": [1.0, 3.0],
                "col3": [float("nan"), float("nan")],
            }
        ),
    )


def test_drop_nan_row_transformer_transform_exclude_columns(frame_row: pl.DataFrame) -> None:
    transformer = DropNanRow(exclude_columns=["col2"])
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": [1.0, float("nan"), 3.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan")],
            }
        ),
    )


def test_drop_nan_row_transformer_transform_empty_row() -> None:
    transformer = DropNanRow()
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_drop_nan_row_transformer_transform_empty() -> None:
    transformer = DropNanRow()
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_drop_nan_row_transformer_transform_missing_policy_ignore(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": [1.0, float("nan"), 3.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan")],
            }
        ),
    )


def test_drop_nan_row_transformer_transform_missing_policy_raise(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_row)


def test_drop_nan_row_transformer_transform_missing_policy_warn(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": [1.0, float("nan"), 3.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan")],
            }
        ),
    )
