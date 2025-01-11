from __future__ import annotations

import warnings
from unittest.mock import patch

import polars as pl
import pytest
from coola import objects_are_equal
from coola.utils import is_numpy_available
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.testing.fixture import sklearn_available
from grizz.transformer import SimpleImputer
from grizz.utils.imports import is_sklearn_available

if is_numpy_available():
    import numpy as np
if is_sklearn_available():
    import sklearn


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, float("nan"), 3, None],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "col3": [10, 20, float("nan"), 40, 50],
            "col4": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Float64, "col2": pl.Float32, "col3": pl.Float32, "col4": pl.String},
    )


##############################################
#     Tests for SimpleImputerTransformer     #
##############################################


@sklearn_available
def test_simple_imputer_transformer_repr() -> None:
    assert repr(SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")) == (
        "SimpleImputerTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', propagate_nulls=True, prefix='', "
        "suffix='_imp')"
    )


@sklearn_available
def test_simple_imputer_transformer_str() -> None:
    assert str(SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")) == (
        "SimpleImputerTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', propagate_nulls=True, prefix='', "
        "suffix='_imp')"
    )


@sklearn_available
def test_simple_imputer_transformer_equal_true() -> None:
    assert SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_columns() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col2", "col3"], prefix="", suffix="_imp")
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_prefix() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="bin_", suffix="_imp")
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_suffix() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="")
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_exclude_columns() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp", exclude_columns=["col4"])
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_exist_policy() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp", exist_policy="warn")
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_missing_policy() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp", missing_policy="warn")
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_propagate_nulls() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp", propagate_nulls=False)
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_kwargs() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp", strategy="mean")
    )


@sklearn_available
def test_simple_imputer_transformer_equal_false_different_type() -> None:
    assert not SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp").equal(42)


@sklearn_available
def test_simple_imputer_transformer_get_args() -> None:
    assert objects_are_equal(
        SimpleImputer(
            columns=["col1", "col3"], prefix="", suffix="_imp", strategy="mean"
        ).get_args(),
        {
            "columns": ("col1", "col3"),
            "prefix": "",
            "suffix": "_imp",
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "propagate_nulls": True,
            "strategy": "mean",
        },
    )


@sklearn_available
def test_simple_imputer_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")
    transformer.fit(dataframe)
    assert transformer._imputer.n_features_in_ == 2


@sklearn_available
def test_simple_imputer_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_imp", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)
    assert transformer._imputer.n_features_in_ == 2


@sklearn_available
def test_simple_imputer_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(columns=["col1", "col3", "col5"], prefix="", suffix="_imp")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


@sklearn_available
def test_simple_imputer_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_imp", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)
    assert transformer._imputer.n_features_in_ == 2


@sklearn_available
def test_simple_imputer_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")
    out = transformer.fit_transform(dataframe)
    assert transformer._imputer.n_features_in_ == 2
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, float("nan"), 3, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, float("nan"), 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_imp": [1.0, 2.0, 2.0, 3.0, None],
                "col3_imp": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float32,
                "col4": pl.String,
                "col1_imp": pl.Float64,
                "col3_imp": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_simple_imputer_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")
    transformer._imputer.fit(
        np.array([[1, 10], [2, 20], [float("nan"), float("nan")], [3, 40], [None, 50]])
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, float("nan"), 3, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, float("nan"), 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_imp": [1.0, 2.0, 2.0, 3.0, None],
                "col3_imp": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float32,
                "col4": pl.String,
                "col1_imp": pl.Float64,
                "col3_imp": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_simple_imputer_transformer_transform_propagate_nulls_true(dataframe: pl.DataFrame) -> None:
    transformer = SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")
    transformer._imputer.fit(
        np.array([[1, 10], [2, 20], [float("nan"), float("nan")], [3, 40], [None, 50]])
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, float("nan"), 3, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, float("nan"), 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_imp": [1.0, 2.0, 2.0, 3.0, None],
                "col3_imp": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float32,
                "col4": pl.String,
                "col1_imp": pl.Float64,
                "col3_imp": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_simple_imputer_transformer_transform_propagate_nulls_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(
        columns=["col1", "col3"], prefix="", suffix="_imp", propagate_nulls=False
    )
    transformer._imputer.fit(
        np.array([[1, 10], [2, 20], [float("nan"), float("nan")], [3, 40], [None, 50]])
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, float("nan"), 3, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, float("nan"), 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_imp": [1.0, 2.0, 2.0, 3.0, 2.0],
                "col3_imp": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float32,
                "col4": pl.String,
                "col1_imp": pl.Float64,
                "col3_imp": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_simple_imputer_transformer_transform_not_fitted(dataframe: pl.DataFrame) -> None:
    transformer = SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")
    with pytest.raises(
        sklearn.exceptions.NotFittedError, match="This SimpleImputer instance is not fitted yet."
    ):
        transformer.transform(dataframe)


@sklearn_available
def test_simple_imputer_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore"
    )
    transformer._imputer.fit(
        np.array([[1, 10], [2, 20], [float("nan"), float("nan")], [3, 40], [None, 50]])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 2.0, 3.0, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10.0, 20.0, 30.0, 40.0, 50.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float64,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_simple_imputer_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_simple_imputer_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn")
    transformer._imputer.fit(
        np.array([[1, 10], [2, 20], [float("nan"), float("nan")], [3, 40], [None, 50]])
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
                "col1": [1.0, 2.0, 2.0, 3.0, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10.0, 20.0, 30.0, 40.0, 50.0],
                "col4": ["a", "b", "c", "d", "e"],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float64,
                "col4": pl.String,
            },
        ),
    )


@sklearn_available
def test_simple_imputer_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_imp", missing_policy="ignore"
    )
    transformer._imputer.fit(
        np.array([[1, 10], [2, 20], [float("nan"), float("nan")], [3, 40], [None, 50]])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, float("nan"), 3, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, float("nan"), 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_imp": [1.0, 2.0, 2.0, 3.0, None],
                "col3_imp": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float32,
                "col4": pl.String,
                "col1_imp": pl.Float64,
                "col3_imp": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_simple_imputer_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(columns=["col1", "col3", "col5"], prefix="", suffix="_imp")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_simple_imputer_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = SimpleImputer(
        columns=["col1", "col3", "col5"], prefix="", suffix="_imp", missing_policy="warn"
    )
    transformer._imputer.fit(
        np.array([[1, 10], [2, 20], [float("nan"), float("nan")], [3, 40], [None, 50]])
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, float("nan"), 3, None],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, float("nan"), 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_imp": [1.0, 2.0, 2.0, 3.0, None],
                "col3_imp": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            schema={
                "col1": pl.Float64,
                "col2": pl.Float32,
                "col3": pl.Float32,
                "col4": pl.String,
                "col1_imp": pl.Float64,
                "col3_imp": pl.Float64,
            },
        ),
    )


def test_simple_imputer_transformer_no_sklearn() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        SimpleImputer(columns=["col1", "col3"], prefix="", suffix="_imp")
