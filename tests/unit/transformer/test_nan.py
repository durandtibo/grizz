from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import DropNanColumn, DropNanRow


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
            "col2": [1.0, float("nan"), 3.0, float("nan"), float("nan")],
            "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
        }
    )


##############################################
#     Tests for DropNanColumnTransformer     #
##############################################


def test_drop_nan_column_transformer_repr() -> None:
    assert repr(DropNanColumn(columns=["col1", "col3"])) == (
        "DropNanColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0)"
    )


def test_drop_nan_column_transformer_repr_with_kwargs() -> None:
    assert repr(DropNanColumn(columns=["col1", "col3"], strict=False)) == (
        "DropNanColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0, strict=False)"
    )


def test_drop_nan_column_transformer_str() -> None:
    assert str(DropNanColumn(columns=["col1", "col3"])) == (
        "DropNanColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0)"
    )


def test_drop_nan_column_transformer_str_with_kwargs() -> None:
    assert str(DropNanColumn(columns=["col1", "col3"], strict=False)) == (
        "DropNanColumnTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', threshold=1.0, strict=False)"
    )


def test_drop_nan_column_transformer_equal_true() -> None:
    assert DropNanColumn(columns=["col1", "col3"]).equal(DropNanColumn(columns=["col1", "col3"]))


def test_drop_nan_column_transformer_equal_false_different_threshold() -> None:
    assert not DropNanColumn(columns=["col1", "col3"]).equal(
        DropNanColumn(columns=["col1", "col3"], threshold=0.5)
    )


def test_drop_nan_column_transformer_equal_false_different_columns() -> None:
    assert not DropNanColumn(columns=["col1", "col3"]).equal(
        DropNanColumn(columns=["col1", "col2", "col3"])
    )


def test_drop_nan_column_transformer_equal_false_different_exclude_columns() -> None:
    assert not DropNanColumn(columns=["col1", "col3"]).equal(
        DropNanColumn(columns=["col1", "col3"], exclude_columns=["col2"])
    )


def test_drop_nan_column_transformer_equal_false_different_missing_policy() -> None:
    assert not DropNanColumn(columns=["col1", "col3"]).equal(
        DropNanColumn(columns=["col1", "col3"], missing_policy="warn")
    )


def test_drop_nan_column_transformer_equal_false_different_kwargs() -> None:
    assert not DropNanColumn(columns=["col1", "col3"]).equal(
        DropNanColumn(columns=["col1", "col3"], characters=None)
    )


def test_drop_nan_column_transformer_equal_false_different_type() -> None:
    assert not DropNanColumn(columns=["col1", "col3"]).equal(42)


def test_drop_nan_column_transformer_get_args() -> None:
    assert objects_are_equal(
        DropNanColumn(columns=["col1", "col3"], strict=False).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "threshold": 1.0,
            "strict": False,
        },
    )


def test_drop_nan_column_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DropNanColumn()
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'DropNanColumnTransformer.fit' as there are no parameters available to fit"
    )


def test_drop_nan_column_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropNanColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_drop_nan_column_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropNanColumn(columns=["col1", "col2", "col5"], threshold=0.4)
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_drop_nan_column_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropNanColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_drop_nan_column_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = DropNanColumn()
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col2": [1.0, float("nan"), 3.0, float("nan"), float("nan")],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


def test_drop_nan_column_transformer_transform_threshold_1(dataframe: pl.DataFrame) -> None:
    transformer = DropNanColumn(threshold=1.0)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col2": [1.0, float("nan"), 3.0, float("nan"), float("nan")],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


def test_drop_nan_column_transformer_transform_threshold_0_4(dataframe: pl.DataFrame) -> None:
    transformer = DropNanColumn(threshold=0.4)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


def test_drop_nan_column_transformer_transform_threshold_0_2(dataframe: pl.DataFrame) -> None:
    transformer = DropNanColumn(threshold=0.2)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


def test_drop_nan_column_transformer_transform_columns(dataframe: pl.DataFrame) -> None:
    transformer = DropNanColumn(columns=["col1", "col2"], threshold=0.4)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


def test_drop_nan_column_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = DropNanColumn(threshold=1.0, exclude_columns=["col3", "col4"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col2": [1.0, float("nan"), 3.0, float("nan"), float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


def test_drop_nan_column_transformer_transform_empty_row() -> None:
    transformer = DropNanColumn(threshold=0.5)
    out = transformer.transform(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_drop_nan_column_transformer_transform_empty() -> None:
    transformer = DropNanColumn(threshold=0.5)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_drop_nan_column_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropNanColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


def test_drop_nan_column_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropNanColumn(columns=["col1", "col2", "col5"], threshold=0.4)
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_drop_nan_column_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropNanColumn(
        columns=["col1", "col2", "col5"], threshold=0.4, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col3": [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            }
        ),
    )


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


def test_drop_nan_row_transformer_equal_true() -> None:
    assert DropNanRow(columns=["col1", "col3"]).equal(DropNanRow(columns=["col1", "col3"]))


def test_drop_nan_row_transformer_equal_false_different_columns() -> None:
    assert not DropNanRow(columns=["col1", "col3"]).equal(
        DropNanRow(columns=["col1", "col2", "col3"])
    )


def test_drop_nan_row_transformer_equal_false_different_exclude_columns() -> None:
    assert not DropNanRow(columns=["col1", "col3"]).equal(
        DropNanRow(columns=["col1", "col3"], exclude_columns=["col2"])
    )


def test_drop_nan_row_transformer_equal_false_different_missing_policy() -> None:
    assert not DropNanRow(columns=["col1", "col3"]).equal(
        DropNanRow(columns=["col1", "col3"], missing_policy="warn")
    )


def test_drop_nan_row_transformer_equal_false_different_type() -> None:
    assert not DropNanRow(columns=["col1", "col3"]).equal(42)


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
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(frame_row)


def test_drop_nan_row_transformer_fit_missing_policy_warn(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
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
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(frame_row)


def test_drop_nan_row_transformer_transform_missing_policy_warn(
    frame_row: pl.DataFrame,
) -> None:
    transformer = DropNanRow(columns=["col1", "col2", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
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
