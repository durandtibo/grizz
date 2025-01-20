from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.transformer import InplaceStripChars, StripChars


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )


###########################################
#     Tests for StripCharsTransformer     #
###########################################


def test_strip_chars_transformer_repr() -> None:
    assert repr(StripChars(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_strip_chars_transformer_repr_with_kwargs() -> None:
    assert repr(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_out", characters=None)
    ) == (
        "StripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', characters=None)"
    )


def test_strip_chars_transformer_str() -> None:
    assert str(StripChars(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_strip_chars_transformer_str_with_kwargs() -> None:
    assert str(StripChars(columns=["col1", "col3"], prefix="", suffix="_out", characters=None)) == (
        "StripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', characters=None)"
    )


def test_strip_chars_transformer_equal_true() -> None:
    assert StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_out")
    )


def test_strip_chars_transformer_equal_false_different_columns() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


def test_strip_chars_transformer_equal_false_different_prefix() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col3"], prefix="p_", suffix="_out")
    )


def test_strip_chars_transformer_equal_false_different_suffix() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_s")
    )


def test_strip_chars_transformer_equal_false_different_exclude_columns() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col2"])
    )


def test_strip_chars_transformer_equal_false_different_exist_policy() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


def test_strip_chars_transformer_equal_false_different_missing_policy() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


def test_strip_chars_transformer_equal_false_different_kwargs() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_out", characters=None)
    )


def test_strip_chars_transformer_equal_false_different_type() -> None:
    assert not StripChars(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


def test_strip_chars_transformer_get_args() -> None:
    assert objects_are_equal(
        StripChars(columns=["col1", "col3"], prefix="", suffix="_out", characters=None).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "characters": None,
        },
    )


def test_strip_chars_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = StripChars(columns=["col1", "col3"], prefix="", suffix="_out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'StripCharsTransformer.fit' as there are no parameters available to fit"
    )


def test_strip_chars_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(
        columns=["col2", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_strip_chars_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StripChars(columns=["col2", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_strip_chars_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(
        columns=["col2", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_strip_chars_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
                "col2_out": ["1", "2", "3", "4", "5"],
                "col3_out": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_strip_chars_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3"], prefix="", suffix="_out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
                "col2_out": ["1", "2", "3", "4", "5"],
                "col3_out": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_strip_chars_transformer_transform_none() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None],
            "col2": ["1", "2", "3", "4", "5", None],
            "col3": ["a ", " b", "  c  ", "d", "e", None],
            "col4": ["a ", " b", "  c  ", "d", "e", None],
        }
    )
    transformer = StripChars(columns=["col2", "col3"], prefix="", suffix="_out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": ["1", "2", "3", "4", "5", None],
                "col3": ["a ", " b", "  c  ", "d", "e", None],
                "col4": ["a ", " b", "  c  ", "d", "e", None],
                "col2_out": ["1", "2", "3", "4", "5", None],
                "col3_out": ["a", "b", "c", "d", "e", None],
            }
        ),
    )


def test_strip_chars_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=None, prefix="", suffix="_out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
                "col2_out": ["1", "2", "3", "4", "5"],
                "col3_out": ["a", "b", "c", "d", "e"],
                "col4_out": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_strip_chars_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(
        columns=None, exclude_columns=["col4", "col5"], prefix="", suffix="_out"
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
                "col2_out": ["1", "2", "3", "4", "5"],
                "col3_out": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_strip_chars_transformer_transform_empty() -> None:
    transformer = StripChars(columns=[], prefix="", suffix="_out")
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_strip_chars_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.String})
    transformer = StripChars(columns=["col"], prefix="", suffix="_out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col": [], "col_out": []}, schema={"col": pl.String, "col_out": pl.String}),
    )


def test_strip_chars_transformer_transform_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3"], prefix="", suffix="", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_strip_chars_transformer_transform_exist_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_strip_chars_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3"], prefix="", suffix="", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
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


def test_strip_chars_transformer_transform_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(
        columns=["col2", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
                "col2_out": ["1", "2", "3", "4", "5"],
                "col3_out": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_strip_chars_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StripChars(columns=["col2", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_strip_chars_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(
        columns=["col2", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a ", " b", "  c  ", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
                "col2_out": ["1", "2", "3", "4", "5"],
                "col3_out": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_strip_chars_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3", "col5"], prefix="", suffix="_out")
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_strip_chars_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=None, prefix="", suffix="_out")
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_strip_chars_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3", "col5"], prefix="", suffix="_out")
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_strip_chars_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=None, prefix="", suffix="_out")
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_strip_chars_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=["col2", "col3", "col5"], prefix="", suffix="_out")
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_strip_chars_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = StripChars(columns=None, prefix="", suffix="_out")
    assert transformer.find_missing_columns(dataframe) == ()


##################################################
#     Tests for InplaceStripCharsTransformer     #
##################################################


def test_inplace_strip_chars_transformer_repr() -> None:
    assert repr(InplaceStripChars(columns=["col1", "col3"])) == (
        "InplaceStripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_strip_chars_transformer_repr_with_kwargs() -> None:
    assert repr(InplaceStripChars(columns=["col1", "col3"], characters=None)) == (
        "InplaceStripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', characters=None)"
    )


def test_inplace_strip_chars_transformer_str() -> None:
    assert str(InplaceStripChars(columns=["col1", "col3"])) == (
        "InplaceStripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_strip_chars_transformer_str_with_kwargs() -> None:
    assert str(InplaceStripChars(columns=["col1", "col3"], characters=None)) == (
        "InplaceStripCharsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', characters=None)"
    )


def test_inplace_strip_chars_transformer_equal_true() -> None:
    assert InplaceStripChars(columns=["col1", "col3"]).equal(
        InplaceStripChars(columns=["col1", "col3"])
    )


def test_inplace_strip_chars_transformer_equal_false_different_columns() -> None:
    assert not InplaceStripChars(columns=["col1", "col3"]).equal(
        InplaceStripChars(columns=["col1", "col2", "col3"])
    )


def test_inplace_strip_chars_transformer_equal_false_different_exclude_columns() -> None:
    assert not InplaceStripChars(columns=["col1", "col3"]).equal(
        InplaceStripChars(columns=["col1", "col3"], exclude_columns=["col2"])
    )


def test_inplace_strip_chars_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceStripChars(columns=["col1", "col3"]).equal(
        InplaceStripChars(columns=["col1", "col3"], missing_policy="warn")
    )


def test_inplace_strip_chars_transformer_equal_false_different_kwargs() -> None:
    assert not InplaceStripChars(columns=["col1", "col3"]).equal(
        InplaceStripChars(columns=["col1", "col3"], characters=None)
    )


def test_inplace_strip_chars_transformer_equal_false_different_type() -> None:
    assert not InplaceStripChars(columns=["col1", "col3"]).equal(42)


def test_inplace_strip_chars_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceStripChars(columns=["col1", "col3"], characters=None).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "characters": None,
        },
    )


def test_inplace_strip_chars_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = InplaceStripChars(columns=["col1", "col3"])
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'InplaceStripCharsTransformer.fit' as there are no parameters available to fit"
    )


def test_inplace_strip_chars_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_inplace_strip_chars_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_inplace_strip_chars_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_inplace_strip_chars_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3"])
    out = transformer.fit_transform(dataframe)
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


def test_inplace_strip_chars_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3"])
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


def test_inplace_strip_chars_transformer_transform_none() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None],
            "col2": ["1", "2", "3", "4", "5", None],
            "col3": ["a ", " b", "  c  ", "d", "e", None],
            "col4": ["a ", " b", "  c  ", "d", "e", None],
        }
    )
    transformer = InplaceStripChars(columns=["col2", "col3"])
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


def test_inplace_strip_chars_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=None)
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


def test_inplace_strip_chars_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=None, exclude_columns=["col4", "col5"])
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


def test_inplace_strip_chars_transformer_transform_empty() -> None:
    transformer = InplaceStripChars(columns=[])
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_inplace_strip_chars_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.String})
    transformer = InplaceStripChars(columns=["col"])
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col": []}, schema={"col": pl.String}),
    )


def test_inplace_strip_chars_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_inplace_strip_chars_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_inplace_strip_chars_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
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


def test_inplace_strip_chars_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"])
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_inplace_strip_chars_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=None)
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_inplace_strip_chars_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"])
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_inplace_strip_chars_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=None)
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_inplace_strip_chars_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=["col2", "col3", "col5"])
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_inplace_strip_chars_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceStripChars(columns=None)
    assert transformer.find_missing_columns(dataframe) == ()
