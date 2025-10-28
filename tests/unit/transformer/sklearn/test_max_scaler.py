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
from grizz.transformer import MaxAbsScaler
from grizz.utils.imports import is_sklearn_available

if is_numpy_available():
    import numpy as np
if is_sklearn_available():
    import sklearn


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


#############################################
#     Tests for MaxAbsScalerTransformer     #
#############################################


@sklearn_available
def test_max_abs_scaler_transformer_repr() -> None:
    assert repr(MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "MaxAbsScalerTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', "
        "propagate_nulls=True)"
    )


@sklearn_available
def test_max_abs_scaler_transformer_str() -> None:
    assert str(MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "MaxAbsScalerTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', "
        "propagate_nulls=True)"
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_true() -> None:
    assert MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_columns() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_prefix() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="bin_", suffix="_out")
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_suffix() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="")
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_exclude_columns() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_exist_policy() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_missing_policy() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_propagate_nulls() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out", propagate_nulls=False)
    )


@sklearn_available
def test_max_abs_scaler_transformer_equal_false_different_type() -> None:
    assert not MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


@sklearn_available
def test_max_abs_scaler_transformer_get_args() -> None:
    assert objects_are_equal(
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out").get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "propagate_nulls": True,
        },
    )


@sklearn_available
def test_max_abs_scaler_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    transformer.fit(dataframe)
    assert objects_are_equal(transformer._scaler.scale_, np.array([5.0, 50.0]))


@sklearn_available
def test_max_abs_scaler_transformer_fit_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = MaxAbsScaler(
        columns=None, prefix="", suffix="_out", exclude_columns=["col2", "col4"]
    )
    transformer.fit(dataframe)
    assert objects_are_equal(transformer._scaler.scale_, np.array([5.0, 50.0]))


@sklearn_available
def test_max_abs_scaler_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)
    assert objects_are_equal(transformer._scaler.scale_, np.array([5.0, 50.0]))


@sklearn_available
def test_max_abs_scaler_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


@sklearn_available
def test_max_abs_scaler_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)
    assert objects_are_equal(transformer._scaler.scale_, np.array([5.0, 50.0]))


@sklearn_available
def test_max_abs_scaler_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
    assert objects_are_equal(transformer._scaler.scale_, np.array([5.0, 50.0]))
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [0.2, 0.4, 0.6, 0.8, 1.0],
                "col3_out": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_max_abs_scaler_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    transformer._scaler.fit(np.array([[5, 50]]))
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [0.2, 0.4, 0.6, 0.8, 1.0],
                "col3_out": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_max_abs_scaler_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = MaxAbsScaler(
        columns=None, prefix="", suffix="_out", exclude_columns=["col2", "col4"]
    )
    transformer._scaler.fit(np.array([[5, 50]]))
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [0.2, 0.4, 0.6, 0.8, 1.0],
                "col3_out": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_max_abs_scaler_transformer_transform_propagate_nulls_true() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0, 1, None, None, 1, float("nan"), float("nan")],
            "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Int64},
    )
    transformer = MaxAbsScaler(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    transformer._scaler.fit(np.array([[5, -2, 50]]))
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
                "col2": [
                    -1.0,
                    -2.0,
                    -3.0,
                    -4.0,
                    -5.0,
                    1,
                    None,
                    None,
                    1,
                    float("nan"),
                    float("nan"),
                ],
                "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                "col1_out": [
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    None,
                    0.2,
                    None,
                    float("nan"),
                    0.2,
                    float("nan"),
                ],
                "col2_out": [
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -2.5,
                    0.5,
                    None,
                    None,
                    0.5,
                    float("nan"),
                    float("nan"),
                ],
                "col3_out": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_max_abs_scaler_transformer_transform_propagate_nulls_false() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0, 1, None, None, 1, float("nan"), float("nan")],
            "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Int64},
    )
    transformer = MaxAbsScaler(
        columns=["col1", "col2", "col3"], prefix="", suffix="_out", propagate_nulls=False
    )
    transformer._scaler.fit(np.array([[5, -2, 50]]))
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
                "col2": [
                    -1.0,
                    -2.0,
                    -3.0,
                    -4.0,
                    -5.0,
                    1,
                    None,
                    None,
                    1,
                    float("nan"),
                    float("nan"),
                ],
                "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                "col1_out": [
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    float("nan"),
                    0.2,
                    float("nan"),
                    float("nan"),
                    0.2,
                    float("nan"),
                ],
                "col2_out": [
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -2.5,
                    0.5,
                    float("nan"),
                    float("nan"),
                    0.5,
                    float("nan"),
                    float("nan"),
                ],
                "col3_out": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_max_abs_scaler_transformer_transform_not_fitted(dataframe: pl.DataFrame) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    with pytest.raises(
        sklearn.exceptions.NotFittedError, match=r"This MaxAbsScaler instance is not fitted yet."
    ):
        transformer.transform(dataframe)


@sklearn_available
def test_max_abs_scaler_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore"
    )
    transformer._scaler.fit(np.array([[5, 50]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [0.2, 0.4, 0.6, 0.8, 1.0],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [0.2, 0.4, 0.6, 0.8, 1.0],
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
def test_max_abs_scaler_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match=r"2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_max_abs_scaler_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn")
    transformer._scaler.fit(np.array([[5, 50]]))
    with pytest.warns(
        ColumnExistsWarning,
        match=r"2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [0.2, 0.4, 0.6, 0.8, 1.0],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [0.2, 0.4, 0.6, 0.8, 1.0],
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
def test_max_abs_scaler_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    transformer._scaler.fit(np.array([[5, 50]]))
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
                "col1_out": [0.2, 0.4, 0.6, 0.8, 1.0],
                "col3_out": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_max_abs_scaler_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_max_abs_scaler_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = MaxAbsScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    transformer._scaler.fit(np.array([[5, 50]]))
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
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
                "col1_out": [0.2, 0.4, 0.6, 0.8, 1.0],
                "col3_out": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.Int64,
                "col4": pl.String,
                "col1_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


def test_max_abs_scaler_transformer_no_sklearn() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'sklearn' package is required but not installed."),
    ):
        MaxAbsScaler(columns=["col1", "col3"], prefix="", suffix="_out")
