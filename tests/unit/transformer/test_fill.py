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
from grizz.transformer import FillNan, FillNull, InplaceFillNan


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, None],
            "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
            "col3": ["a", "b", "c", "d", None],
            "col4": [1.2, float("nan"), 3.2, None, 5.2],
        },
        schema={
            "col1": pl.Int64,
            "col2": pl.Float64,
            "col3": pl.String,
            "col4": pl.Float64,
        },
    )


########################################
#     Tests for FillNanTransformer     #
########################################


def test_fill_nan_transformer_repr() -> None:
    assert repr(FillNan(columns=["col1", "col4"], prefix="", suffix="_out")) == (
        "FillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_fill_nan_transformer_repr_with_kwargs() -> None:
    assert repr(FillNan(columns=["col1", "col4"], prefix="", suffix="_out", value=100)) == (
        "FillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', value=100)"
    )


def test_fill_nan_transformer_str() -> None:
    assert str(FillNan(columns=["col1", "col4"], prefix="", suffix="_out")) == (
        "FillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_fill_nan_transformer_str_with_kwargs() -> None:
    assert str(FillNan(columns=["col1", "col4"], prefix="", suffix="_out", value=100)) == (
        "FillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', value=100)"
    )


def test_fill_nan_transformer_equal_true() -> None:
    assert FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col3"], prefix="", suffix="_out")
    )


def test_fill_nan_transformer_equal_false_different_columns() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


def test_fill_nan_transformer_equal_false_different_prefix() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col3"], prefix="p_", suffix="_out")
    )


def test_fill_nan_transformer_equal_false_different_suffix() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col3"], prefix="", suffix="_s")
    )


def test_fill_nan_transformer_equal_false_different_exclude_columns() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


def test_fill_nan_transformer_equal_false_different_exist_policy() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


def test_fill_nan_transformer_equal_false_different_missing_policy() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


def test_fill_nan_transformer_equal_false_different_kwargs() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNan(columns=["col1", "col3"], prefix="", suffix="_out", value=100)
    )


def test_fill_nan_transformer_equal_false_different_type() -> None:
    assert not FillNan(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


def test_fill_nan_transformer_get_args() -> None:
    assert objects_are_equal(
        FillNan(columns=["col1", "col3"], prefix="", suffix="_out", value=100).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "value": 100,
        },
    )


def test_fill_nan_transformer_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = FillNan(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'FillNanTransformer.fit' as there are no parameters available to fit"
    )


def test_fill_nan_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(
        columns=["col1", "col4", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="ignore",
        value=100,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_fill_nan_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], prefix="", suffix="_out", value=100)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_fill_nan_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(
        columns=["col1", "col4", "col5"], prefix="", suffix="_out", missing_policy="warn", value=100
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_fill_nan_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col4_out": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col4_out": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=None, prefix="", suffix="_out", value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col2_out": [1.2, 2.2, 3.2, 4.2, 100.0],
                "col4_out": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col2_out": pl.Float64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(
        columns=None, prefix="", suffix="_out", value=100, exclude_columns=["col2", "col3", "col5"]
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col4_out": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_transform_empty() -> None:
    transformer = FillNan(columns=[], prefix="", suffix="_out", value=100)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_fill_nan_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.Float64})
    transformer = FillNan(columns=["col"], prefix="", suffix="_out", value=100)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col": [], "col_out": []}, schema={"col": pl.Float64, "col_out": pl.Float64}),
    )


def test_fill_nan_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNan(
        columns=["col1", "col4"], prefix="", suffix="", exist_policy="ignore", value=100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
        ),
    )


def test_fill_nan_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNan(columns=["col1", "col4"], prefix="", suffix="", value=100)
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_fill_nan_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNan(
        columns=["col1", "col4"], prefix="", suffix="", exist_policy="warn", value=100
    )
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
        ),
    )


def test_fill_nan_transformer_transform_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(
        columns=["col1", "col4", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="ignore",
        value=100,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col4_out": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], prefix="", suffix="_out", value=100)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_fill_nan_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(
        columns=["col1", "col4", "col5"], prefix="", suffix="_out", missing_policy="warn", value=100
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col4_out": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], prefix="", suffix="", value=100)
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_fill_nan_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=None, prefix="", suffix="", value=100)
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_nan_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], prefix="", suffix="", value=100)
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_fill_nan_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=None, prefix="", suffix="", value=100)
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_nan_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], prefix="", suffix="", value=100)
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_fill_nan_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=None, prefix="", suffix="", value=100)
    assert transformer.find_missing_columns(dataframe) == ()


###############################################
#     Tests for InplaceFillNanTransformer     #
###############################################


def test_inplace_fill_nan_transformer_repr() -> None:
    assert repr(InplaceFillNan(columns=["col1", "col4"])) == (
        "InplaceFillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_fill_nan_transformer_repr_with_kwargs() -> None:
    assert repr(InplaceFillNan(columns=["col1", "col4"], value=100)) == (
        "InplaceFillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "missing_policy='raise', value=100)"
    )


def test_inplace_fill_nan_transformer_str() -> None:
    assert str(InplaceFillNan(columns=["col1", "col4"])) == (
        "InplaceFillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "missing_policy='raise')"
    )


def test_inplace_fill_nan_transformer_str_with_kwargs() -> None:
    assert str(InplaceFillNan(columns=["col1", "col4"], value=100)) == (
        "InplaceFillNanTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "missing_policy='raise', value=100)"
    )


def test_inplace_fill_nan_transformer_equal_true() -> None:
    assert InplaceFillNan(columns=["col1", "col3"]).equal(InplaceFillNan(columns=["col1", "col3"]))


def test_inplace_fill_nan_transformer_equal_false_different_columns() -> None:
    assert not InplaceFillNan(columns=["col1", "col3"]).equal(
        InplaceFillNan(columns=["col1", "col2", "col3"])
    )


def test_inplace_fill_nan_transformer_equal_false_different_exclude_columns() -> None:
    assert not InplaceFillNan(columns=["col1", "col3"]).equal(
        InplaceFillNan(columns=["col1", "col3"], exclude_columns=["col4"])
    )


def test_inplace_fill_nan_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceFillNan(columns=["col1", "col3"]).equal(
        InplaceFillNan(columns=["col1", "col3"], missing_policy="warn")
    )


def test_inplace_fill_nan_transformer_equal_false_different_kwargs() -> None:
    assert not InplaceFillNan(columns=["col1", "col3"]).equal(
        InplaceFillNan(columns=["col1", "col3"], value=100)
    )


def test_inplace_fill_nan_transformer_equal_false_different_type() -> None:
    assert not InplaceFillNan(columns=["col1", "col3"]).equal(42)


def test_inplace_fill_nan_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceFillNan(columns=["col1", "col3"], value=100).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "value": 100,
        },
    )


def test_inplace_fill_nan_transformer_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = InplaceFillNan(columns=["col1", "col4"], value=100)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'InplaceFillNanTransformer.fit' as there are no parameters available to fit"
    )


def test_inplace_fill_nan_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(
        columns=["col1", "col4", "col5"], missing_policy="ignore", value=100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_inplace_fill_nan_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFillNan(columns=["col2", "col3", "col5"], value=100)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_inplace_fill_nan_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=["col1", "col4", "col5"], missing_policy="warn", value=100)
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_inplace_fill_nan_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=["col1", "col4"], value=100)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_inplace_fill_nan_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=["col1", "col4"], value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_inplace_fill_nan_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=None, value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, 100.0],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_inplace_fill_nan_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=None, value=100, exclude_columns=["col2", "col3", "col5"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
        ),
    )


def test_inplace_fill_nan_transformer_transform_empty() -> None:
    transformer = InplaceFillNan(columns=[], value=100)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_inplace_fill_nan_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.Float64})
    transformer = InplaceFillNan(columns=["col"], value=100)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col": []}, schema={"col": pl.Float64}),
    )


def test_inplace_fill_nan_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFillNan(
        columns=["col1", "col4", "col5"], missing_policy="ignore", value=100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
        ),
    )


def test_inplace_fill_nan_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFillNan(columns=["col2", "col3", "col5"], value=100)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_inplace_fill_nan_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = InplaceFillNan(columns=["col1", "col4", "col5"], missing_policy="warn", value=100)
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
        ),
    )


def test_inplace_fill_nan_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_inplace_fill_nan_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=None, value=100)
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_inplace_fill_nan_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_inplace_fill_nan_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=None, value=100)
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_inplace_fill_nan_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_inplace_fill_nan_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = InplaceFillNan(columns=None, value=100)
    assert transformer.find_missing_columns(dataframe) == ()


#########################################
#     Tests for FillNullTransformer     #
#########################################


def test_fill_null_transformer_repr() -> None:
    assert repr(FillNull(columns=["col1", "col4"], prefix="", suffix="_out")) == (
        "FillNullTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_fill_null_transformer_repr_with_kwargs() -> None:
    assert repr(FillNull(columns=["col1", "col4"], prefix="", suffix="_out", value=100)) == (
        "FillNullTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', value=100)"
    )


def test_fill_null_transformer_str() -> None:
    assert str(FillNull(columns=["col1", "col4"], prefix="", suffix="_out")) == (
        "FillNullTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')"
    )


def test_fill_null_transformer_str_with_kwargs() -> None:
    assert str(FillNull(columns=["col1", "col4"], prefix="", suffix="_out", value=100)) == (
        "FillNullTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', value=100)"
    )


def test_fill_null_transformer_equal_true() -> None:
    assert FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col3"], prefix="", suffix="_out")
    )


def test_fill_null_transformer_equal_false_different_columns() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


def test_fill_null_transformer_equal_false_different_prefix() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col3"], prefix="p_", suffix="_out")
    )


def test_fill_null_transformer_equal_false_different_suffix() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col3"], prefix="", suffix="_s")
    )


def test_fill_null_transformer_equal_false_different_exclude_columns() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


def test_fill_null_transformer_equal_false_different_exist_policy() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


def test_fill_null_transformer_equal_false_different_missing_policy() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


def test_fill_null_transformer_equal_false_different_kwargs() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        FillNull(columns=["col1", "col3"], prefix="", suffix="_out", value=100)
    )


def test_fill_null_transformer_equal_false_different_type() -> None:
    assert not FillNull(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


def test_fill_null_transformer_get_args() -> None:
    assert objects_are_equal(
        FillNull(columns=["col1", "col3"], prefix="", suffix="_out", value=100).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "value": 100,
        },
    )


def test_fill_null_transformer_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = FillNull(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'FillNullTransformer.fit' as there are no parameters available to fit"
    )


def test_fill_null_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(
        columns=["col1", "col4", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="ignore",
        value=100,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_fill_null_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], prefix="", suffix="_out", value=100)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_fill_null_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(
        columns=["col1", "col4", "col5"], prefix="", suffix="_out", missing_policy="warn", value=100
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_fill_null_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col1_out": [1, 2, 3, 4, 100],
                "col4_out": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col1_out": pl.Int64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col1", "col4"], prefix="", suffix="_out", value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col1_out": [1, 2, 3, 4, 100],
                "col4_out": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col1_out": pl.Int64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=None, prefix="", suffix="_out", value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col1_out": [1, 2, 3, 4, 100],
                "col2_out": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3_out": ["a", "b", "c", "d", "100"],
                "col4_out": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col1_out": pl.Int64,
                "col2_out": pl.Float64,
                "col3_out": pl.String,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(
        columns=None, prefix="", suffix="_out", value=100, exclude_columns=["col2", "col3", "col5"]
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col1_out": [1, 2, 3, 4, 100],
                "col4_out": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col1_out": pl.Int64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_transform_empty() -> None:
    transformer = FillNull(columns=[], prefix="", suffix="_out", value=100)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_fill_null_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.String})
    transformer = FillNull(columns=["col"], prefix="", suffix="_out", value=100)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col": [], "col_out": []}, schema={"col": pl.String, "col_out": pl.String}),
    )


def test_fill_null_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNull(
        columns=["col1", "col4"], prefix="", suffix="", exist_policy="ignore", value=100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 100],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
        ),
    )


def test_fill_null_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNull(columns=["col1", "col4"], prefix="", suffix="", value=100)
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_fill_null_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNull(
        columns=["col1", "col4"], prefix="", suffix="", exist_policy="warn", value=100
    )
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 100],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
        ),
    )


def test_fill_null_transformer_transform_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(
        columns=["col1", "col4", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="ignore",
        value=100,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col1_out": [1, 2, 3, 4, 100],
                "col4_out": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col1_out": pl.Int64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], prefix="", suffix="_out", value=100)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_fill_null_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(
        columns=["col1", "col4", "col5"], prefix="", suffix="_out", missing_policy="warn", value=100
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
                "col1_out": [1, 2, 3, 4, 100],
                "col4_out": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
                "col1_out": pl.Int64,
                "col4_out": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], prefix="", suffix="", value=100)
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_fill_null_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=None, prefix="", suffix="", value=100)
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_null_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], prefix="", suffix="", value=100)
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_fill_null_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=None, prefix="", suffix="", value=100)
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_null_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], prefix="", suffix="", value=100)
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_fill_null_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=None, prefix="", suffix="", value=100)
    assert transformer.find_missing_columns(dataframe) == ()
