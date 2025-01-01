from __future__ import annotations

import operator
import warnings
from unittest.mock import patch

import polars as pl
import pytest
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
    assert repr(OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")) == (
        "OrdinalEncoderTransformer(columns=('col1', 'col2', 'col3'), prefix='', suffix='_ord', "
        "exclude_columns=(), propagate_nulls=True, exist_policy='raise', missing_policy='raise')"
    )


@sklearn_available
def test_ordinal_encoder_transformer_str() -> None:
    assert str(OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")) == (
        "OrdinalEncoderTransformer(columns=('col1', 'col2', 'col3'), prefix='', suffix='_ord', "
        "exclude_columns=(), propagate_nulls=True, exist_policy='raise', missing_policy='raise')"
    )


@sklearn_available
def test_ordinal_encoder_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")
    transformer.fit(dataframe)
    assert transformer._encoder.n_features_in_ == 3


@sklearn_available
def test_ordinal_encoder_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3", "col5"],
        prefix="",
        suffix="_ord",
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
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_ord")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


@sklearn_available
def test_ordinal_encoder_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_ord", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)
    assert transformer._encoder.n_features_in_ == 3


@sklearn_available
def test_ordinal_encoder_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")
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
                "col1_ord": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_ord": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_ord": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_ord": pl.Float64,
                "col2_ord": pl.Float64,
                "col3_ord": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")
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
                "col1_ord": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_ord": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_ord": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_ord": pl.Float64,
                "col2_ord": pl.Float64,
                "col3_ord": pl.Float64,
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
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")
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
                "col1_ord": [
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
                "col2_ord": [
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
                "col3_ord": [
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
                "col1_ord": pl.Float64,
                "col2_ord": pl.Float64,
                "col3_ord": pl.Float64,
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
        columns=["col1", "col2", "col3"], prefix="", suffix="_ord", propagate_nulls=False
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
                "col1_ord": [
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
                "col2_ord": [
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
                "col3_ord": [
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
                "col1_ord": pl.Float64,
                "col2_ord": pl.Float64,
                "col3_ord": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_not_fitted(dataframe: pl.DataFrame) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")
    # Use a if-else structure because the exception depends on the sklearn version
    if compare_version("scikit-learn", operator.ge, "1.4.0"):
        exception = pytest.raises(
            sklearn.exceptions.NotFittedError,
            match="This OrdinalEncoder instance is not fitted yet.",
        )
    else:
        exception = pytest.raises(
            AttributeError, match="'OrdinalEncoder' object has no attribute '_missing_indices'"
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
    with pytest.raises(ColumnExistsError, match="3 columns already exist in the DataFrame:"):
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
        match="3 columns already exist in the DataFrame and will be overwritten:",
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
        suffix="_ord",
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
                "col1_ord": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_ord": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_ord": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_ord": pl.Float64,
                "col2_ord": pl.Float64,
                "col3_ord": pl.Float64,
            },
        ),
    )


@sklearn_available
def test_ordinal_encoder_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_ord")
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


@sklearn_available
def test_ordinal_encoder_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = OrdinalEncoder(
        columns=["col1", "col2", "col3", "col5"], prefix="", suffix="_ord", missing_policy="warn"
    )
    transformer._encoder.fit(
        [[4, -1.0, "c"], [5, -2.0, "a"], [1, -3.0, "d"], [2, -4.0, "b"], [3, -5.0, "e"]]
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
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
                "col1_ord": [3.0, 4.0, 0.0, 1.0, 2.0],
                "col2_ord": [4.0, 3.0, 2.0, 1.0, 0.0],
                "col3_ord": [2.0, 0.0, 3.0, 1.0, 4.0],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float32,
                "col3": pl.String,
                "col4": pl.Int64,
                "col1_ord": pl.Float64,
                "col2_ord": pl.Float64,
                "col3_ord": pl.Float64,
            },
        ),
    )


def test_ordinal_encoder_transformer_no_sklearn() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        OrdinalEncoder(columns=["col1", "col2", "col3"], prefix="", suffix="_ord")
