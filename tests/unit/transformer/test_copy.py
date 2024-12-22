from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import CopyColumns


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


############################################
#     Tests for CopyColumnsTransformer     #
############################################


def test_copy_columns_transformer_repr() -> None:
    assert repr(CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")) == (
        "CopyColumnsTransformer(columns=('col1', 'col3'), prefix='p_', suffix='_s', ignore_missing=False)"
    )


def test_copy_columns_transformer_str() -> None:
    assert str(CopyColumns(columns=["col1", "col3"], prefix="p_", suffix="_s")) == (
        "CopyColumnsTransformer(columns=('col1', 'col3'), prefix='p_', suffix='_s', ignore_missing=False)"
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


def test_copy_columns_transformer_transform_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CopyColumns(columns=["col1", "col3", "col5"], prefix="p_", suffix="_s")
    with pytest.raises(ColumnNotFoundError, match="1 columns are missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_copy_columns_transformer_transform_ignore_missing_true(dataframe: pl.DataFrame) -> None:
    transformer = CopyColumns(
        columns=["col1", "col3", "col5"], prefix="p_", suffix="_s", ignore_missing=True
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 columns are missing in the DataFrame and will be ignored:"
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
