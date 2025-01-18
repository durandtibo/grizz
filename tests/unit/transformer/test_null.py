from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import DropNullColumn, DropNullRow

###############################################
#     Tests for DropNullColumnTransformer     #
###############################################


@pytest.fixture
def frame_col() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )


def test_drop_null_column_transformer_repr() -> None:
    assert repr(DropNullColumn(columns=["col1", "col3"])) == (
        "DropNullColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0)"
    )


def test_drop_null_column_transformer_repr_with_kwargs() -> None:
    assert repr(DropNullColumn(columns=["col1", "col3"], strict=False)) == (
        "DropNullColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0, strict=False)"
    )


def test_drop_null_column_transformer_str() -> None:
    assert str(DropNullColumn(columns=["col1", "col3"])) == (
        "DropNullColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0)"
    )


def test_drop_null_column_transformer_str_with_kwargs() -> None:
    assert str(DropNullColumn(columns=["col1", "col3"], strict=False)) == (
        "DropNullColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0, strict=False)"
    )


def test_drop_null_column_transformer_equal_true() -> None:
    assert DropNullColumn(columns=["col1", "col3"]).equal(DropNullColumn(columns=["col1", "col3"]))


def test_drop_null_column_transformer_equal_false_different_threshold() -> None:
    assert not DropNullColumn(columns=["col1", "col3"]).equal(
        DropNullColumn(columns=["col1", "col3"], threshold=0.5)
    )


def test_drop_null_column_transformer_equal_false_different_columns() -> None:
    assert not DropNullColumn(columns=["col1", "col3"]).equal(
        DropNullColumn(columns=["col1", "col2", "col3"])
    )


def test_drop_null_column_transformer_equal_false_different_exclude_columns() -> None:
    assert not DropNullColumn(columns=["col1", "col3"]).equal(
        DropNullColumn(columns=["col1", "col3"], exclude_columns=["col2"])
    )


def test_drop_null_column_transformer_equal_false_different_missing_policy() -> None:
    assert not DropNullColumn(columns=["col1", "col3"]).equal(
        DropNullColumn(columns=["col1", "col3"], missing_policy="warn")
    )


def test_drop_null_column_transformer_equal_false_different_kwargs() -> None:
    assert not DropNullColumn(columns=["col1", "col3"]).equal(
        DropNullColumn(columns=["col1", "col3"], characters=None)
    )


def test_drop_null_column_transformer_equal_false_different_type() -> None:
    assert not DropNullColumn(columns=["col1", "col3"]).equal(42)


def test_drop_null_column_transformer_get_args() -> None:
    assert objects_are_equal(
        DropNullColumn(columns=["col1", "col3"], strict=False).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "threshold": 1.0,
            "strict": False,
        },
    )


def test_drop_null_column_transformer_fit(
    frame_col: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DropNullColumn()
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_col)
    assert caplog.messages[0].startswith(
        "Skipping 'DropNullColumnTransformer.fit' as there are no parameters available to fit"
    )


def test_drop_null_column_transformer_fit_missing_policy_ignore(
    frame_col: pl.DataFrame,
) -> None:
    transformer = DropNullColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(frame_col)


def test_drop_null_column_transformer_fit_missing_policy_raise(
    frame_col: pl.DataFrame,
) -> None:
    transformer = DropNullColumn(columns=["col1", "col2", "col5"], threshold=0.4)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(frame_col)


def test_drop_null_column_transformer_fit_missing_policy_warn(
    frame_col: pl.DataFrame,
) -> None:
    transformer = DropNullColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(frame_col)


def test_drop_null_column_transformer_fit_transform(frame_col: pl.DataFrame) -> None:
    transformer = DropNullColumn()
    out = transformer.fit_transform(frame_col)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_drop_null_column_transformer_transform_threshold_1(frame_col: pl.DataFrame) -> None:
    transformer = DropNullColumn(threshold=1.0)
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


def test_drop_null_column_transformer_transform_threshold_0_4(frame_col: pl.DataFrame) -> None:
    transformer = DropNullColumn(threshold=0.4)
    out = transformer.transform(frame_col)
    assert_frame_equal(
        out, pl.DataFrame({"col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None]})
    )


def test_drop_null_column_transformer_transform_threshold_0_2(frame_col: pl.DataFrame) -> None:
    transformer = DropNullColumn(threshold=0.2)
    out = transformer.transform(frame_col)
    assert_frame_equal(out, pl.DataFrame({}))


def test_drop_null_column_transformer_transform_columns(frame_col: pl.DataFrame) -> None:
    transformer = DropNullColumn(columns=["col1", "col2"], threshold=0.4)
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


def test_drop_null_column_transformer_transform_exclude_columns(frame_col: pl.DataFrame) -> None:
    transformer = DropNullColumn(threshold=1.0, exclude_columns=["col3", "col4"])
    out = transformer.transform(frame_col)
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


def test_drop_null_column_transformer_transform_empty_row() -> None:
    transformer = DropNullColumn(threshold=0.5)
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_drop_null_column_transformer_transform_empty() -> None:
    transformer = DropNullColumn(threshold=0.5)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_drop_null_column_transformer_transform_missing_policy_ignore(
    frame_col: pl.DataFrame,
) -> None:
    transformer = DropNullColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_drop_null_column_transformer_transform_missing_policy_raise(
    frame_col: pl.DataFrame,
) -> None:
    transformer = DropNullColumn(columns=["col1", "col2", "col5"], threshold=0.4)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_col)


def test_drop_null_column_transformer_transform_missing_policy_warn(
    frame_col: pl.DataFrame,
) -> None:
    transformer = DropNullColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
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


############################################
#     Tests for DropNullRowTransformer     #
############################################


@pytest.fixture
def frame_row() -> pl.DataFrame:
    return pl.DataFrame(
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
    frame_row: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DropNullRow()
    with caplog.at_level(logging.INFO):
        transformer.fit(frame_row)
    assert caplog.messages[0].startswith(
        "Skipping 'DropNullRowTransformer.fit' as there are no parameters available to fit"
    )


def test_drop_null_row_transformer_fit_missing_policy_ignore(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(frame_row)


def test_drop_null_row_transformer_fit_missing_policy_raise(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(frame_row)


def test_drop_null_row_transformer_fit_missing_policy_warn(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(frame_row)


def test_drop_null_row_transformer_fit_transform(frame_row: pl.DataFrame) -> None:
    transformer = DropNullRow()
    out = transformer.fit_transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform(frame_row: pl.DataFrame) -> None:
    transformer = DropNullRow()
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_columns(frame_row: pl.DataFrame) -> None:
    transformer = DropNullRow(columns=["col2", "col3"])
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-31"],
                "col2": [1, 3],
                "col3": [None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_exclude_columns(frame_row: pl.DataFrame) -> None:
    transformer = DropNullRow(exclude_columns=["col2"])
    out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_empty_row() -> None:
    transformer = DropNullRow()
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_drop_null_row_transformer_transform_empty() -> None:
    transformer = DropNullRow()
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_drop_null_row_transformer_transform_missing_policy_ignore(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )


def test_drop_null_row_transformer_transform_missing_policy_raise(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(frame_row)


def test_drop_null_row_transformer_transform_missing_policy_warn(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNullRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(frame_row)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-31", "2020-12-31"],
                "col2": [1, 3, None],
                "col3": [None, None, None],
            }
        ),
    )
