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
from grizz.transformer import StandardScaler
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


###############################################
#     Tests for StandardScalerTransformer     #
###############################################


@sklearn_available
def test_standard_scaler_transformer_repr() -> None:
    assert repr(StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StandardScalerTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', propagate_nulls=True, prefix='', "
        "suffix='_out')"
    )


@sklearn_available
def test_standard_scaler_transformer_str() -> None:
    assert str(StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")) == (
        "StandardScalerTransformer(columns=('col1', 'col3'), exclude_columns=(), "
        "missing_policy='raise', exist_policy='raise', propagate_nulls=True, prefix='', "
        "suffix='_out')"
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_true() -> None:
    assert StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_columns() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_prefix() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="bin_", suffix="_out")
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_suffix() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="")
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_exclude_columns() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_exist_policy() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_missing_policy() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_propagate_nulls() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out", propagate_nulls=False)
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_kwargs() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out", with_std=False)
    )


@sklearn_available
def test_standard_scaler_transformer_transformer_equal_false_different_type() -> None:
    assert not StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


@sklearn_available
def test_standard_scaler_transformer_transformer_get_args() -> None:
    assert objects_are_equal(
        StandardScaler(
            columns=["col1", "col3"], prefix="", suffix="_out", with_std=False
        ).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "propagate_nulls": True,
            "with_std": False,
        },
    )


@sklearn_available
def test_standard_scaler_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    transformer.fit(dataframe)
    assert transformer._scaler.n_features_in_ == 2


@sklearn_available
def test_standard_scaler_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)
    assert transformer._scaler.n_features_in_ == 2


@sklearn_available
def test_standard_scaler_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


@sklearn_available
def test_standard_scaler_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)
    assert transformer._scaler.n_features_in_ == 2


@sklearn_available
def test_standard_scaler_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
    assert transformer._scaler.n_features_in_ == 2
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
                "col3_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
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
def test_standard_scaler_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    transformer._scaler.fit(np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]))
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [10, 20, 30, 40, 50],
                "col4": ["a", "b", "c", "d", "e"],
                "col1_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
                "col3_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
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
def test_standard_scaler_transformer_transform_propagate_nulls_true() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0, 1, None, None, 1, float("nan"), float("nan")],
            "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Int64},
    )
    transformer = StandardScaler(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    transformer._scaler.fit(
        np.array([[1, -1, 10], [2, -2, 20], [3, -3, 30], [4, -4, 40], [5, -5, 50]])
    )
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
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                    None,
                    -1.414213562373095,
                    None,
                    float("nan"),
                    -1.414213562373095,
                    float("nan"),
                ],
                "col2_out": [
                    1.414213562373095,
                    0.7071067811865475,
                    0.0,
                    -0.7071067811865475,
                    -1.414213562373095,
                    2.82842712474619,
                    None,
                    None,
                    2.82842712474619,
                    float("nan"),
                    float("nan"),
                ],
                "col3_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                    2.1213203435596424,
                    2.82842712474619,
                    3.5355339059327373,
                    4.242640687119285,
                    4.949747468305833,
                    5.65685424949238,
                ],
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
def test_standard_scaler_transformer_transform_propagate_nulls_false() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0, 1, None, None, 1, float("nan"), float("nan")],
            "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Int64},
    )
    transformer = StandardScaler(
        columns=["col1", "col2", "col3"], prefix="", suffix="_out", propagate_nulls=False
    )
    transformer._scaler.fit(
        np.array([[1, -1, 10], [2, -2, 20], [3, -3, 30], [4, -4, 40], [5, -5, 50]])
    )
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
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                    float("nan"),
                    -1.414213562373095,
                    float("nan"),
                    float("nan"),
                    -1.414213562373095,
                    float("nan"),
                ],
                "col2_out": [
                    1.414213562373095,
                    0.7071067811865475,
                    0.0,
                    -0.7071067811865475,
                    -1.414213562373095,
                    2.82842712474619,
                    float("nan"),
                    float("nan"),
                    2.82842712474619,
                    float("nan"),
                    float("nan"),
                ],
                "col3_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                    2.1213203435596424,
                    2.82842712474619,
                    3.5355339059327373,
                    4.242640687119285,
                    4.949747468305833,
                    5.65685424949238,
                ],
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
def test_standard_scaler_transformer_transform_not_fitted(dataframe: pl.DataFrame) -> None:
    transformer = StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    with pytest.raises(
        sklearn.exceptions.NotFittedError, match="This StandardScaler instance is not fitted yet."
    ):
        transformer.transform(dataframe)


@sklearn_available
def test_standard_scaler_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="ignore"
    )
    transformer._scaler.fit(np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
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
def test_standard_scaler_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(columns=["col1", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match="2 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_standard_scaler_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(
        columns=["col1", "col3"], prefix="", suffix="", exist_policy="warn"
    )
    transformer._scaler.fit(np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]))
    with pytest.warns(
        ColumnExistsWarning,
        match="2 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
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
def test_standard_scaler_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="ignore"
    )
    transformer._scaler.fit(np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]))
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
                "col1_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
                "col3_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
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
def test_standard_scaler_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(columns=["col1", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_standard_scaler_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = StandardScaler(
        columns=["col1", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    transformer._scaler.fit(np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]))
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
                "col1_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
                "col3_out": [
                    -1.414213562373095,
                    -0.7071067811865475,
                    0.0,
                    0.7071067811865475,
                    1.414213562373095,
                ],
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


def test_standard_scaler_transformer_no_sklearn() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        StandardScaler(columns=["col1", "col3"], prefix="", suffix="_out")
