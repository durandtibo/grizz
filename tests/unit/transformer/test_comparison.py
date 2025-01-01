from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.transformer import (
    Equal,
    EqualMissing,
    Greater,
    GreaterEqual,
    Lower,
    LowerEqual,
    NotEqual,
    NotEqualMissing,
)


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "col3": [10, 20, 30, 40, 50],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Float32, "col3": pl.Int64, "col4": pl.String},
    )


######################################
#     Tests for EqualTransformer     #
######################################


def test_equal_transformer_repr() -> None:
    assert repr(Equal(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "EqualTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_equal_transformer_str() -> None:
    assert str(Equal(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "EqualTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_equal_transformer_fit(dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    transformer = Equal(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'EqualTransformer.fit' as there are no parameters available to fit"
    )


def test_equal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = Equal(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Equal(columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = Equal(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Equal(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = Equal(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_transformer_transform_target_2(dataframe: pl.DataFrame) -> None:
    transformer = Equal(columns=["col1", "col3"], target=2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, True, False, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = Equal(columns=["col1", "col2"], target=0.0, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [False, None, False, None, False, False, False],
                "col2_ind": [False, False, None, None, False, False, False],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Equal(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [False, False, True, False, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Equal(columns=["col1", "col3"], target=3, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Equal(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="warn"
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
                "col1": [False, False, True, False, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Equal(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Equal(columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Equal(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


#############################################
#     Tests for EqualMissingTransformer     #
#############################################


def test_equal_missing_transformer_repr() -> None:
    assert repr(EqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "EqualMissingTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_equal_missing_transformer_str() -> None:
    assert str(EqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "EqualMissingTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_equal_missing_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = EqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'EqualMissingTransformer.fit' as there are no parameters available to fit"
    )


def test_equal_missing_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = EqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_equal_missing_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = EqualMissing(columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_equal_missing_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = EqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_equal_missing_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = EqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_missing_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = EqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_missing_transformer_transform_target_2(dataframe: pl.DataFrame) -> None:
    transformer = EqualMissing(columns=["col1", "col3"], target=2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, True, False, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_missing_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = EqualMissing(columns=["col1", "col2"], target=None, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [False, True, False, True, False, False, False],
                "col2_ind": [False, False, True, True, False, False, False],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_equal_missing_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = EqualMissing(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [False, False, True, False, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_equal_missing_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = EqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_equal_missing_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = EqualMissing(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="warn"
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
                "col1": [False, False, True, False, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_equal_missing_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = EqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_equal_missing_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = EqualMissing(columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_equal_missing_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = EqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


#############################################
#     Tests for GreaterEqualTransformer     #
#############################################


def test_greater_equal_transformer_repr() -> None:
    assert repr(GreaterEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "GreaterEqualTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_greater_equal_transformer_str() -> None:
    assert str(GreaterEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "GreaterEqualTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_greater_equal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = GreaterEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'GreaterEqualTransformer.fit' as there are no parameters available to fit"
    )


def test_greater_equal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_greater_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind"
    )
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_greater_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_greater_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = GreaterEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = GreaterEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_equal_transformer_transform_target_3(dataframe: pl.DataFrame) -> None:
    transformer = GreaterEqual(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, True, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = GreaterEqual(columns=["col1", "col2"], target=0.0, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [True, None, True, None, True, False, True],
                "col2_ind": [False, True, None, None, False, True, True],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_greater_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [False, False, False, False, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_greater_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = GreaterEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_greater_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="warn"
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
                "col1": [False, False, False, False, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_greater_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind"
    )
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_greater_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = GreaterEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


########################################
#     Tests for GreaterTransformer     #
########################################


def test_greater_transformer_repr() -> None:
    assert repr(Greater(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "GreaterTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_greater_transformer_str() -> None:
    assert str(Greater(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "GreaterTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_greater_transformer_fit(dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    transformer = Greater(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'GreaterTransformer.fit' as there are no parameters available to fit"
    )


def test_greater_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = Greater(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_greater_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Greater(columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_greater_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = Greater(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_greater_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Greater(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = Greater(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_transformer_transform_target_3(dataframe: pl.DataFrame) -> None:
    transformer = Greater(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = Greater(columns=["col1", "col2"], target=0.0, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [True, None, True, None, True, False, True],
                "col2_ind": [False, True, None, None, False, True, True],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_greater_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Greater(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [False, False, False, False, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_greater_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Greater(columns=["col1", "col3"], target=4.2, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_greater_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Greater(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="warn"
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
                "col1": [False, False, False, False, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_greater_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Greater(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_greater_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Greater(columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_greater_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Greater(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [False, False, False, False, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


###########################################
#     Tests for LowerEqualTransformer     #
###########################################


def test_lower_equal_transformer_repr() -> None:
    assert repr(LowerEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "LowerEqualTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_lower_equal_transformer_str() -> None:
    assert str(LowerEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "LowerEqualTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_lower_equal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = LowerEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'LowerEqualTransformer.fit' as there are no parameters available to fit"
    )


def test_lower_equal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = LowerEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_lower_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LowerEqual(columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_lower_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = LowerEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_lower_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = LowerEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = LowerEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_equal_transformer_transform_target_3(dataframe: pl.DataFrame) -> None:
    transformer = LowerEqual(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = LowerEqual(columns=["col1", "col2"], target=0.0, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [False, None, False, None, False, True, False],
                "col2_ind": [True, False, None, None, True, False, False],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_lower_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LowerEqual(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [True, True, True, True, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_lower_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LowerEqual(columns=["col1", "col3"], target=4.2, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_lower_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LowerEqual(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="warn"
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
                "col1": [True, True, True, True, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_lower_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LowerEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LowerEqual(columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_lower_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = LowerEqual(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


######################################
#     Tests for LowerTransformer     #
######################################


def test_lower_transformer_repr() -> None:
    assert repr(Lower(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "LowerTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_lower_transformer_str() -> None:
    assert str(Lower(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")) == (
        "LowerTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_lower_transformer_fit(dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    transformer = Lower(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'LowerTransformer.fit' as there are no parameters available to fit"
    )


def test_lower_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = Lower(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_lower_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Lower(columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_lower_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = Lower(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_lower_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Lower(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = Lower(columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_transformer_transform_target_3(dataframe: pl.DataFrame) -> None:
    transformer = Lower(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, False, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = Lower(columns=["col1", "col2"], target=0.0, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [False, None, False, None, False, True, False],
                "col2_ind": [True, False, None, None, True, False, False],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_lower_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Lower(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [True, True, True, True, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_lower_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Lower(columns=["col1", "col3"], target=4.2, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_lower_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Lower(
        columns=["col1", "col3"], target=4.2, prefix="", suffix="", exist_policy="warn"
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
                "col1": [True, True, True, True, False],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [False, False, False, False, False],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_lower_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Lower(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_lower_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Lower(columns=["col1", "col3", "col5"], target=4.2, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_lower_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Lower(
        columns=["col1", "col3", "col5"],
        target=4.2,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, True, True, False],
                "col3_ind": [False, False, False, False, False],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


#########################################
#     Tests for NotEqualTransformer     #
#########################################


def test_not_equal_transformer_repr() -> None:
    assert repr(NotEqual(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "NotEqualTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_not_equal_transformer_str() -> None:
    assert str(NotEqual(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "NotEqualTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_not_equal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = NotEqual(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'NotEqualTransformer.fit' as there are no parameters available to fit"
    )


def test_not_equal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = NotEqual(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_not_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqual(columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_not_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = NotEqual(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_not_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = NotEqual(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = NotEqual(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_transformer_transform_target_2(dataframe: pl.DataFrame) -> None:
    transformer = NotEqual(columns=["col1", "col3"], target=2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, False, True, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = NotEqual(columns=["col1", "col2"], target=0.0, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [True, None, True, None, True, True, True],
                "col2_ind": [True, True, None, None, True, True, True],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqual(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [True, True, False, True, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_not_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqual(columns=["col1", "col3"], target=3, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_not_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqual(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="warn"
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
                "col1": [True, True, False, True, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_not_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqual(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqual(columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_not_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqual(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


################################################
#     Tests for NotEqualMissingTransformer     #
################################################


def test_not_equal_missing_transformer_repr() -> None:
    assert repr(NotEqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "NotEqualMissingTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_not_equal_missing_transformer_str() -> None:
    assert str(NotEqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")) == (
        "NotEqualMissingTransformer(columns=('col1', 'col3'), target=3, prefix='', suffix='_ind', "
        "exclude_columns=(), exist_policy='raise', missing_policy='raise')"
    )


def test_not_equal_missing_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = NotEqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'NotEqualMissingTransformer.fit' as there are no parameters available to fit"
    )


def test_not_equal_missing_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_not_equal_missing_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind"
    )
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_not_equal_missing_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_not_equal_missing_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = NotEqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_missing_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = NotEqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_missing_transformer_transform_target_2(dataframe: pl.DataFrame) -> None:
    transformer = NotEqualMissing(columns=["col1", "col3"], target=2, prefix="", suffix="_ind")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, False, True, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_missing_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = NotEqualMissing(columns=["col1", "col2"], target=None, prefix="", suffix="_ind")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [-1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "col1_ind": [True, False, True, False, True, True, True],
                "col2_ind": [True, True, False, False, True, True, True],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col1_ind": pl.Boolean,
                "col2_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_missing_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [True, True, False, True, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_not_equal_missing_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqualMissing(columns=["col1", "col3"], target=3, prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


def test_not_equal_missing_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3"], target=3, prefix="", suffix="", exist_policy="warn"
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
                "col1": [True, True, False, True, True],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [True, True, True, True, True],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Boolean,
                "col2": pl.Float32,
                "col3": pl.Boolean,
                "col4": pl.String,
            },
        ),
    )


def test_not_equal_missing_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )


def test_not_equal_missing_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3", "col5"], target=3, prefix="", suffix="_ind"
    )
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_not_equal_missing_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = NotEqualMissing(
        columns=["col1", "col3", "col5"],
        target=3,
        prefix="",
        suffix="_ind",
        missing_policy="warn",
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
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_ind": [True, True, False, True, True],
                "col3_ind": [True, True, True, True, True],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_ind": pl.Boolean,
                "col3_ind": pl.Boolean,
            },
        ),
    )
