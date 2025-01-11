from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
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
    assert (
        repr(DropDuplicate())
        == "DropDuplicateTransformer(columns=None, exclude_columns=(), missing_policy='raise')"
    )


def test_drop_duplicate_transformer_repr_with_kwargs() -> None:
    assert repr(DropDuplicate(columns=["col1", "col3"], keep="first")) == (
        "DropDuplicateTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', keep='first')"
    )


def test_drop_duplicate_transformer_str() -> None:
    assert (
        str(DropDuplicate())
        == "DropDuplicateTransformer(columns=None, exclude_columns=(), missing_policy='raise')"
    )


def test_drop_duplicate_transformer_str_with_kwargs() -> None:
    assert str(DropDuplicate(columns=["col1", "col3"], keep="first")) == (
        "DropDuplicateTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', keep='first')"
    )


def test_drop_duplicate_transformer_equal_true() -> None:
    assert DropDuplicate(columns=["col1", "col3"]).equal(DropDuplicate(columns=["col1", "col3"]))


def test_drop_duplicate_transformer_equal_false_different_columns() -> None:
    assert not DropDuplicate(columns=["col1", "col3"]).equal(
        DropDuplicate(columns=["col1", "col2", "col3"])
    )


def test_drop_duplicate_transformer_equal_false_different_exclude_columns() -> None:
    assert not DropDuplicate(columns=["col1", "col3"]).equal(
        DropDuplicate(columns=["col1", "col3"], exclude_columns=["col4"])
    )


def test_drop_duplicate_transformer_equal_false_different_exist_policy() -> None:
    assert not DropDuplicate(columns=["col1", "col3"]).equal(
        DropDuplicate(columns=["col1", "col3"], exist_policy="warn")
    )


def test_drop_duplicate_transformer_equal_false_different_missing_policy() -> None:
    assert not DropDuplicate(columns=["col1", "col3"]).equal(
        DropDuplicate(columns=["col1", "col3"], missing_policy="warn")
    )


def test_drop_duplicate_transformer_equal_false_different_kwargs() -> None:
    assert not DropDuplicate(columns=["col1", "col3"]).equal(
        DropDuplicate(columns=["col1", "col3"], threshold=1.0)
    )


def test_drop_duplicate_transformer_equal_false_different_type() -> None:
    assert not DropDuplicate(columns=["col1", "col3"]).equal(42)


def test_drop_duplicate_transformer_get_args() -> None:
    assert objects_are_equal(
        DropDuplicate(columns=["col1", "col3"], keep="first").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "keep": "first",
        },
    )


def test_drop_duplicate_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = DropDuplicate(keep="first", maintain_order=True)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'DropDuplicateTransformer.fit' as there are no parameters available to fit"
    )


def test_drop_duplicate_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropDuplicate(
        columns=["col1", "col2", "col5"], keep="first", maintain_order=True, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_drop_duplicate_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropDuplicate(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_drop_duplicate_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropDuplicate(
        columns=["col1", "col2", "col5"], keep="first", maintain_order=True, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_drop_duplicate_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = DropDuplicate(keep="first", maintain_order=True)
    out = transformer.fit_transform(dataframe)
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
                "col2": ["1", "2", "3"],
                "col3": ["1", "2", "3"],
                "col4": ["a", "a", "a"],
            }
        ),
    )


def test_drop_duplicate_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = DropDuplicate(
        exclude_columns=["col1", "col2", "col5"], keep="first", maintain_order=True
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["1", "2", "3"],
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


def test_drop_duplicate_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropDuplicate(
        columns=["col1", "col2", "col5"], keep="first", maintain_order=True, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_drop_duplicate_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropDuplicate(columns=["col1", "col2", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_drop_duplicate_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = DropDuplicate(
        columns=["col1", "col2", "col5"], keep="first", maintain_order=True, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
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
