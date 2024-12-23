from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import FilterCardinality


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1, 1, 1, 1, 1],
            "col3": ["a", "b", "c", "a", "b"],
            "col4": [1.2, float("nan"), 3.2, None, 5.2],
        }
    )


##################################################
#     Tests for FilterCardinalityTransformer     #
##################################################


def test_filter_cardinality_transformer_repr() -> None:
    assert repr(FilterCardinality(columns=["col1", "col2", "col3"], n_min=2, n_max=5)).startswith(
        "FilterCardinalityTransformer("
    )


def test_filter_cardinality_transformer_str() -> None:
    assert str(FilterCardinality(columns=["col1", "col2", "col3"], n_min=2, n_max=5)).startswith(
        "FilterCardinalityTransformer("
    )


def test_filter_cardinality_transformer_fit(
    caplog: pytest.LogCaptureFixture, dataframe: pl.DataFrame
) -> None:
    transformer = FilterCardinality(columns=["col1", "col2", "col3"], n_min=2, n_max=5)
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'FilterCardinalityTransformer.fit' as there are no parameters available to fit"
    )


def test_filter_cardinality_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = FilterCardinality(columns=["col1", "col2", "col3"], n_min=2, n_max=5)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col3": ["a", "b", "c", "a", "b"],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
            }
        ),
    )


def test_filter_cardinality_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = FilterCardinality(columns=["col1", "col2", "col3"], n_min=2, n_max=5)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col3": ["a", "b", "c", "a", "b"],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
            }
        ),
    )


def test_filter_cardinality_transformer_transform_default(dataframe: pl.DataFrame) -> None:
    transformer = FilterCardinality()
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 1, 1, 1, 1],
                "col3": ["a", "b", "c", "a", "b"],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
            }
        ),
    )


def test_filter_cardinality_transformer_transform_min_1() -> None:
    transformer = FilterCardinality(n_min=1)
    frame = pl.DataFrame(
        {
            "col1": [1, 1, 1, 1, 1],
            "col2": [1, 1, 2, 2, 2],
            "col3": ["a", "b", "c", "a", "b"],
            "col4": [1.2, float("nan"), 3.2, None, 3.2],
        }
    )
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 1, 1, 1, 1],
                "col2": [1, 1, 2, 2, 2],
                "col3": ["a", "b", "c", "a", "b"],
                "col4": [1.2, float("nan"), 3.2, None, 3.2],
            }
        ),
    )


def test_filter_cardinality_transformer_transform_min_2() -> None:
    transformer = FilterCardinality(n_min=2)
    frame = pl.DataFrame(
        {
            "col1": [1, 1, 1, 1, 1],
            "col2": [1, 1, 2, 2, 2],
            "col3": ["a", "b", "c", "a", "b"],
            "col4": [1.2, float("nan"), 3.2, None, 3.2],
        }
    )
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col2": [1, 1, 2, 2, 2],
                "col3": ["a", "b", "c", "a", "b"],
                "col4": [1.2, float("nan"), 3.2, None, 3.2],
            }
        ),
    )


def test_filter_cardinality_transformer_transform_max_3() -> None:
    transformer = FilterCardinality(n_max=3)
    frame = pl.DataFrame(
        {
            "col1": [1, 1, 1, 1, 1],
            "col2": [1, 1, 2, 2, 2],
            "col3": ["a", "b", "c", "a", "b"],
            "col4": [1.2, float("nan"), 3.2, None, 3.2],
        }
    )
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 1, 1, 1, 1],
                "col2": [1, 1, 2, 2, 2],
            }
        ),
    )


def test_filter_cardinality_transformer_transform_empty_rows() -> None:
    frame = pl.DataFrame({"col1": [], "col2": [], "col3": []})
    transformer = FilterCardinality(columns=["col1", "col2", "col3"], n_min=2, n_max=5)
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({}))


def test_filter_cardinality_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FilterCardinality(
        columns=["col1", "col2", "col3", "col5"], n_min=2, n_max=5, missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col3": ["a", "b", "c", "a", "b"],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
            }
        ),
    )


def test_filter_cardinality_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FilterCardinality(columns=["col1", "col2", "col3", "col5"], n_min=2, n_max=5)
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_filter_cardinality_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FilterCardinality(
        columns=["col1", "col2", "col3", "col5"], n_min=2, n_max=5, missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col3": ["a", "b", "c", "a", "b"],
                "col4": [1.2, float("nan"), 3.2, None, 5.2],
            }
        ),
    )
