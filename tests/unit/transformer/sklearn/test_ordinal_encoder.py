from __future__ import annotations

import operator
import warnings
from unittest.mock import patch

import polars as pl
import pytest
from coola import objects_are_equal
from coola.utils import is_numpy_available
from feu import compare_version
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.testing.fixture import sklearn_available
from grizz.transformer import OrdinalEncoder
from grizz.utils.imports import is_sklearn_available

if is_numpy_available():
    pass
if is_sklearn_available():
    import sklearn


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [4, 5, 1, 2, 3],
            "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "col3": ["c", "a", "d", "b", "e"],
            "col4": [10, 20, 30, 40, 50],
        },
        schema={"col1": pl.Int64, "col2": pl.Float32, "col3": pl.String, "col4": pl.Int64},
    )


###############################################
#     Tests for OrdinalEncoderTransformer     #
###############################################


@sklearn_available
def test_ordinal_encoder_transformer_repr() -> None:
    assert repr(OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")) == (
        "OrdinalEncoderTransformer(columns=('col1', 'col2', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', "
        "propagate_nulls=True)"
    )


@sklearn_available
def test_ordinal_encoder_transformer_str() -> None:
    assert str(OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")) == (
        "OrdinalEncoderTransformer(columns=('col1', 'col2', 'col3'), exclude_columns=(), "
        "exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', "
        "propagate_nulls=True)"
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_true() -> None:
    assert OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out")
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_columns() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_prefix() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="bin_", suffix="_out")
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_suffix() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="")
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_exclude_columns() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out", exclude_columns=["col4"])
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_exist_policy() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out", exist_policy="warn")
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_missing_policy() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out", missing_policy="warn")
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_propagate_nulls() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out", propagate_nulls=False)
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_kwargs() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(
        OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out", max_categories=10)
    )


@sklearn_available
def test_ordinal_encoder_transformer_equal_false_different_type() -> None:
    assert not OrdinalEncoder(columns=["col1", "col3"], prefix="", suffix="_out").equal(42)


@sklearn_available
def test_ordinal_encoder_transformer_get_args() -> None:
    assert objects_are_equal(
        OrdinalEncoder(
            columns=["col1", "col3"], prefix="", suffix="_out", max_categories=10
        ).get_args(),
        {
            "columns": ("col1", "col3"),
            "exclude_columns": (),
            "exist_policy": "raise",
            "missing_policy": "raise",
            "prefix": "",
            "suffix": "_out",
            "propagate_nulls": True,
            "max_categories": 10,
        },
    )


@sklearn_available
def test_ordinal_encoder_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    transformer.fit(dataframe)
    assert transformer._encoder.n_features_in_ == 3


@sklearn_available
def test_ordinal_encoder_transformer_fit_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(
        columns=None, prefix="", suffix="_out", exclude_columns=["col2", "col4"]
    )
    transformer.fit(dataframe)
    assert transformer._encoder.n_features_in_ == 2


@sklearn_available
def test_ordinal_encoder_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)
    assert transformer._encoder.n_features_in_ == 3


@sklearn_available
def test_ordinal_encoder_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


@sklearn_available
def test_ordinal_encoder_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)
    assert transformer._encoder.n_features_in_ == 3


@sklearn_available
def test_ordinal_encoder_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    out = transformer.fit_transform(dataframe)
    assert transformer._encoder.n_features_in_ == 3
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [4, 5, 1, 2, 3],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": ["c", "a", "d", "b", "e"],
                "col4": [10, 20, 30, 40, 50],
                "col1_out": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_out": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_out": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    transformer._encoder.fit(
        [[4, -1.0, "c"], [5, -2.0, "a"], [1, -3.0, "d"], [2, -4.0, "b"], [3, -5.0, "e"]]
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [4, 5, 1, 2, 3],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": ["c", "a", "d", "b", "e"],
                "col4": [10, 20, 30, 40, 50],
                "col1_out": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_out": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_out": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=None, prefix="", suffix="_out", exclude_columns=["col4"])
    transformer._encoder.fit(
        [[4, -1.0, "c"], [5, -2.0, "a"], [1, -3.0, "d"], [2, -4.0, "b"], [3, -5.0, "e"]]
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [4, 5, 1, 2, 3],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": ["c", "a", "d", "b", "e"],
                "col4": [10, 20, 30, 40, 50],
                "col1_out": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_out": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_out": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_propagate_nulls_true() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
                -1.0,
                None,
                None,
                -1.0,
                float("nan"),
                float("nan"),
            ],
            "col3": ["a", "b", "c", "d", "e", None, None, None, None, None, None],
        },
        schema={"col1": pl.Float32, "col2": pl.Float64, "col3": pl.String},
    )
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    transformer._encoder.fit(
        [
            [4, -1.0, "c"],
            [5, -2.0, "a"],
            [1, -3.0, "d"],
            [2, -4.0, "b"],
            [3, -5.0, "e"],
            [float("nan"), float("nan"), None],
        ]
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
                    -1.0,
                    None,
                    None,
                    -1.0,
                    float("nan"),
                    float("nan"),
                ],
                "col3": ["a", "b", "c", "d", "e", None, None, None, None, None, None],
                "col1_out": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    None,
                    0.0,
                    None,
                    float("nan"),
                    0.0,
                    float("nan"),
                ],
                "col2_out": [
                    4.0,
                    3.0,
                    2.0,
                    1.0,
                    0.0,
                    4.0,
                    None,
                    None,
                    4.0,
                    float("nan"),
                    float("nan"),
                ],
                "col3_out": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float64,
                "col3": pl.String,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_propagate_nulls_false() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, None, 1, None, float("nan"), 1, float("nan")],
            "col2": [
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
                -1.0,
                None,
                None,
                -1.0,
                float("nan"),
                float("nan"),
            ],
            "col3": ["a", "b", "c", "d", "e", None, None, None, None, None, None],
        },
        schema={"col1": pl.Float32, "col2": pl.Float64, "col3": pl.String},
    )
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3"], prefix="", suffix="_out", propagate_nulls=False
    )
    transformer._encoder.fit(
        [
            [4, -1.0, "c"],
            [5, -2.0, "a"],
            [1, -3.0, "d"],
            [2, -4.0, "b"],
            [3, -5.0, "e"],
            [float("nan"), float("nan"), None],
        ]
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
                    -1.0,
                    None,
                    None,
                    -1.0,
                    float("nan"),
                    float("nan"),
                ],
                "col3": ["a", "b", "c", "d", "e", None, None, None, None, None, None],
                "col1_out": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    float("nan"),
                    0.0,
                    float("nan"),
                    float("nan"),
                    0.0,
                    float("nan"),
                ],
                "col2_out": [
                    4.0,
                    3.0,
                    2.0,
                    1.0,
                    0.0,
                    4.0,
                    float("nan"),
                    float("nan"),
                    4.0,
                    float("nan"),
                    float("nan"),
                ],
                "col3_out": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float64,
                "col3": pl.String,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_not_fitted(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
    # Use a if-else structure because the exception depends on the sklearn version
    if compare_version("scikit-learn", operator.ge, "1.4.0"):
        exception = pytest.raises(
            sklearn.exceptions.NotFittedError,
            match=r"This OrdinalEncoder instance is not fitted yet.",
        )
    else:
        exception = pytest.raises(
            AttributeError, match=r"'OrdinalEncoder' object has no attribute '_missing_indices'"
        )
    with exception:
        transformer.transform(dataframe)


@sklearn_available
def test_ordinal_encoder_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3"], prefix="", suffix="", exist_policy="ignore"
    )
    transformer._encoder.fit(
        [[4, -1.0, "c"], [5, -2.0, "a"], [1, -3.0, "d"], [2, -4.0, "b"], [3, -5.0, "e"]]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3": [2.0, 0.0, 3.0, 1.0, 4.0],
                "col4": [10, 20, 30, 40, 50],
            },
            schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64, "col4": pl.Int64},
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="")
    with pytest.raises(ColumnExistsError, match=r"3 columns already exist in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_ordinal_encoder_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3"], prefix="", suffix="", exist_policy="warn"
    )
    transformer._encoder.fit(
        [[4, -1.0, "c"], [5, -2.0, "a"], [1, -3.0, "d"], [2, -4.0, "b"], [3, -5.0, "e"]]
    )
    with pytest.warns(
        ColumnExistsWarning,
        match=r"3 columns already exist in the DataFrame and will be overwritten:",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3": [2.0, 0.0, 3.0, 1.0, 4.0],
                "col4": [10, 20, 30, 40, 50],
            },
            schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64, "col4": pl.Int64},
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3", "col5"],
        prefix="",
        suffix="_out",
        missing_policy="ignore",
    )
    transformer._encoder.fit(
        [[4, -1.0, "c"], [5, -2.0, "a"], [1, -3.0, "d"], [2, -4.0, "b"], [3, -5.0, "e"]]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [4, 5, 1, 2, 3],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": ["c", "a", "d", "b", "e"],
                "col4": [10, 20, 30, 40, 50],
                "col1_out": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_out": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_out": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_out")
    with pytest.raises(ColumnNotFoundError, match=r"1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_ordinal_encoder_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_out", missing_policy="warn"
    )
    transformer._encoder.fit(
        [[4, -1.0, "c"], [5, -2.0, "a"], [1, -3.0, "d"], [2, -4.0, "b"], [3, -5.0, "e"]]
    )
    with pytest.warns(
        ColumnNotFoundWarning, match=r"1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [4, 5, 1, 2, 3],
                "col2": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "col3": ["c", "a", "d", "b", "e"],
                "col4": [10, 20, 30, 40, 50],
                "col1_out": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_out": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_out": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_out": pl.Float64,
                "col2_out": pl.Float64,
                "col3_out": pl.Float64,
            },
        ),
    )


def test_ordinal_encoder_transformer_no_sklearn() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'sklearn' package is required but not installed."),
    ):
        OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_out")
