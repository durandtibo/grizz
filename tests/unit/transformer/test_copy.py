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
from grizz.transformer import CopyColumn, CopyColumns


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
    )


###########################################
#     Tests for CopyColumnTransformer     #
###########################################


def test_copy_column_transformer_repr() -> None:
    assert repr(CopyColumn(in_col="col1", out_col="out")) == (
        "CopyColumnTransformer(in_col='col1', out_col='out', exist_policy='raise', "
        "missing_policy='raise')"
    )


def test_copy_column_transformer_str() -> None:
    assert str(CopyColumn(in_col="col1", out_col="out")) == (
        "CopyColumnTransformer(in_col='col1', out_col='out', exist_policy='raise', "
        "missing_policy='raise')"
    )


def test_copy_column_transformer_equal_true() -> None:
    assert CopyColumn(in_col="col1", out_col="out").equal(CopyColumn(in_col="col1", out_col="out"))


def test_copy_column_transformer_equal_false_different_in_col() -> None:
    assert not CopyColumn(in_col="col1", out_col="out").equal(
        CopyColumn(in_col="col", out_col="out")
    )


def test_copy_column_transformer_equal_false_different_out_col() -> None:
    assert not CopyColumn(in_col="col1", out_col="out").equal(
        CopyColumn(in_col="col1", out_col="col")
    )


def test_copy_column_transformer_equal_false_different_exist_policy() -> None:
    assert not CopyColumn(in_col="col1", out_col="out").equal(
        CopyColumn(in_col="col1", out_col="out", exist_policy="warn")
    )


def test_copy_column_transformer_equal_false_different_missing_policy() -> None:
    assert not CopyColumn(in_col="col1", out_col="out").equal(
        CopyColumn(in_col="col1", out_col="out", missing_policy="warn")
    )


def test_copy_column_transformer_equal_false_different_type() -> None:
    assert not CopyColumn(in_col="col1", out_col="out").equal(42)


def test_copy_column_transformer_get_args() -> None:
    assert objects_are_equal(
        CopyColumn(in_col="col1", out_col="out").get_args(),
        {
            "in_col": "col1",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_copy_column_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = CopyColumn(in_col="col1", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'CopyColumnTransformer.fit' as there are no parameters available to fit"
    )


def test_copy_column_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumn(in_col="in", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_copy_column_transformer_fit_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumn(in_col="in", out_col="out")
    with pytest.raises(ColumnNotFoundError, match=r"column 'in' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_copy_column_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumn(in_col="in", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'in' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_copy_column_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumn(in_col="col1", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [1, 2, 3, 4, 5],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_copy_column_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumn(in_col="col1", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "out": [1, 2, 3, 4, 5],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "out": pl.Int64,
            },
        ),
    )


def test_copy_column_transformer_transform_empty() -> None:
    transformer = CopyColumn(in_col="col1", out_col="out")
    out = transformer.transform(
        pl.DataFrame(
            data={"col1": [], "col2": [], "col3": [], "col4": []},
            schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={"col1": [], "col2": [], "col3": [], "col4": [], "out": []},
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "out": pl.String,
            },
        ),
    )


def test_copy_column_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumn(in_col="col1", out_col="col2", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_copy_column_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumn(in_col="col1", out_col="col2")
    with pytest.raises(ColumnExistsError, match=r"column 'col2' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_copy_column_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumn(in_col="col1", out_col="col2", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match=r"column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_copy_column_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumn(in_col="in", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_copy_column_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumn(in_col="in", out_col="out")
    with pytest.raises(ColumnNotFoundError, match=r"column 'in' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_copy_column_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumn(in_col="in", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match=r"column 'in' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


############################################
#     Tests for CopyColumnsTransformer     #
############################################


def test_copy_columns_transformer_repr() -> None:
    assert repr(CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")) == (
        "CopyColumnsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='p_', suffix='_s')"
    )


def test_copy_columns_transformer_str() -> None:
    assert str(CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")) == (
        "CopyColumnsTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='p_', suffix='_s')"
    )


def test_copy_columns_transformer_equal_true() -> None:
    assert CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(
        CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")
    )


def test_copy_columns_transformer_equal_false_prefix() -> None:
    assert not CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(
        CopyColumns(columns=["col1", "col3"], prefix="prefix_", suffix="_s")
    )


def test_copy_columns_transformer_equal_false_suffix() -> None:
    assert not CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(
        CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_suffix")
    )


def test_copy_columns_transformer_equal_false_different_columns() -> None:
    assert not CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(
        CopyColumns(columns=["col1", "col2", "col3"], prefix="p_", suffix="_s")
    )


def test_copy_columns_transformer_equal_false_different_exclude_columns() -> None:
    assert not CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(
        CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s", exclude_columns=["col2"])
    )


def test_copy_columns_transformer_equal_false_different_exist_policy() -> None:
    assert not CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(
        CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s", exist_policy="warn")
    )


def test_copy_columns_transformer_equal_false_different_missing_policy() -> None:
    assert not CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(
        CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s", missing_policy="warn")
    )


def test_copy_columns_transformer_equal_false_different_type() -> None:
    assert not CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").equal(42)


def test_copy_columns_transformer_get_args() -> None:
    assert objects_are_equal(
        CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s").get_args(),
        {
            "columns": ("col1", "col3"),
            "prefix": "p_",
            "suffix": "_s",
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_copy_columns_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'CopyColumnsTransformer.fit' as there are no parameters available to fit"
    )


def test_copy_columns_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(
        columns=["col1", "col3", "col5"], prefix="p_", suffix="_s", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_copy_columns_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumns(columns=["col1", "col3", "col5"], prefix="p_", suffix="_s")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_copy_columns_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(
        columns=["col1", "col3", "col5"], prefix="p_", suffix="_s", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_copy_columns_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "p_col1_s": [1, 2, 3, 4, 5],
                "p_col3_s": ["1", "2", "3", "4", "5"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "p_col1_s": pl.Int64,
                "p_col3_s": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "p_col1_s": [1, 2, 3, 4, 5],
                "p_col3_s": ["1", "2", "3", "4", "5"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "p_col1_s": pl.Int64,
                "p_col3_s": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform_empty() -> None:
    transformer = CopyColumns(columns=["col1", "col2"], prefix="p_", suffix="_s")
    out = transformer.transform(
        pl.DataFrame(
            data={"col1": [], "col2": [], "col3": [], "col4": []},
            schema={"col1": pl.String, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            data={"col1": [], "col2": [], "col3": [], "col4": [], "p_col1_s": [], "p_col2_s": []},
            schema={
                "col1": pl.String,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "p_col1_s": pl.String,
                "p_col2_s": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform_columns_none() -> None:
    transformer = CopyColumns(columns=None, prefix="p_", suffix="_s")
    out = transformer.transform(
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
                "p_col1_s": [1, 2, 3, 4, 5],
                "p_col2_s": ["1", "2", "3", "4", "5"],
                "p_col3_s": ["1", "2", "3", "4", "5"],
                "p_col4_s": ["101", "102", "103", "104", "105"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "p_col1_s": pl.Int64,
                "p_col2_s": pl.String,
                "p_col3_s": pl.String,
                "p_col4_s": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform_columns_empty() -> None:
    transformer = CopyColumns(columns=[], prefix="p_", suffix="_s")
    out = transformer.transform(
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        ),
    )


def test_copy_columns_transformer_transform_exclude_columns() -> None:
    transformer = CopyColumns(
        columns=None, exclude_columns=["col4", "col5"], prefix="p_", suffix="_s"
    )
    out = transformer.transform(
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
            },
            schema={"col1": pl.Int64, "col2": pl.String, "col3": pl.String, "col4": pl.String},
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["101", "102", "103", "104", "105"],
                "p_col1_s": [1, 2, 3, 4, 5],
                "p_col2_s": ["1", "2", "3", "4", "5"],
                "p_col3_s": ["1", "2", "3", "4", "5"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "p_col1_s": pl.Int64,
                "p_col2_s": pl.String,
                "p_col3_s": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform_exist_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumns(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match=r"2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_copy_columns_transformer_transform_exist_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match=r"2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(
        columns=["col1", "col3", "col5"], prefix="p_", suffix="_s", missing_policy="ignore"
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
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "p_col1_s": [1, 2, 3, 4, 5],
                "p_col3_s": ["1", "2", "3", "4", "5"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "p_col1_s": pl.Int64,
                "p_col3_s": pl.String,
            },
        ),
    )


def test_copy_columns_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumns(columns=["col1", "col3", "col5"], prefix="p_", suffix="_s")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_copy_columns_transformer_transform_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(
        columns=["col1", "col3", "col5"], prefix="p_", suffix="_s", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
                "p_col1_s": [1, 2, 3, 4, 5],
                "p_col3_s": ["1", "2", "3", "4", "5"],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.String,
                "col3": pl.String,
                "col4": pl.String,
                "p_col1_s": pl.Int64,
                "p_col3_s": pl.String,
            },
        ),
    )
