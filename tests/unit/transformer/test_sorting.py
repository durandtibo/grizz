from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer import Sort, SortColumns


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [None, 1, 2, None],
            "col2": [None, 6.0, 5.0, 4.0],
            "col3": [None, "a", "c", "b"],
        }
    )


#####################################
#     Tests for SortTransformer     #
#####################################


def test_sort_transformer_repr() -> None:
    assert (
        repr(Sort(columns=["col3", "col1"]))
        == "SortTransformer(columns=('col3', 'col1'), exclude_columns=(), missing_policy='raise')"
    )


def test_sort_transformer_repr_kwargs() -> None:
    assert (
        repr(Sort(columns=["col3", "col1"], descending=True))
        == "SortTransformer(columns=('col3', 'col1'), exclude_columns=(), "
        "missing_policy='raise', descending=True)"
    )


def test_sort_transformer_str() -> None:
    assert (
        str(Sort(columns=["col3", "col1"]))
        == "SortTransformer(columns=('col3', 'col1'), exclude_columns=(), missing_policy='raise')"
    )


def test_sort_transformer_str_kwargs() -> None:
    assert (
        str(Sort(columns=["col3", "col1"], descending=True))
        == "SortTransformer(columns=('col3', 'col1'), exclude_columns=(), "
        "missing_policy='raise', descending=True)"
    )


def test_sort_transformer_fit(caplog: pytest.LogCaptureFixture, dataframe: pl.DataFrame) -> None:
    transformer = Sort(columns=["col3", "col1"])
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'SortTransformer.fit' as there are no parameters available to fit"
    )


def test_sort_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Sort(columns=["col3", "col1", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_sort_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Sort(columns=["col3", "col1", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.fit(dataframe)


def test_sort_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Sort(columns=["col3", "col1", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        transformer.fit(dataframe)


def test_sort_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = Sort(columns=["col3", "col1"])
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 1, None, 2],
                "col2": [None, 6.0, 4.0, 5.0],
                "col3": [None, "a", "b", "c"],
            }
        ),
    )


def test_sort_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = Sort(columns=["col3", "col1"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 1, None, 2],
                "col2": [None, 6.0, 4.0, 5.0],
                "col3": [None, "a", "b", "c"],
            }
        ),
    )


def test_sort_transformer_transform_null_last(dataframe: pl.DataFrame) -> None:
    transformer = Sort(columns=["col3", "col1"], nulls_last=True)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, None, 2, None],
                "col2": [6.0, 4.0, 5.0, None],
                "col3": ["a", "b", "c", None],
            }
        ),
    )


def test_sort_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = Sort()
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, None, 1, 2],
                "col2": [None, 4.0, 6.0, 5.0],
                "col3": [None, "b", "a", "c"],
            }
        ),
    )


def test_sort_transformer_transform_exclude_columns(dataframe: pl.DataFrame) -> None:
    transformer = Sort(exclude_columns=["col1", "col4"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, None, 2, 1],
                "col2": [None, 4.0, 5.0, 6.0],
                "col3": [None, "b", "c", "a"],
            }
        ),
    )


def test_sort_transformer_transform_empty_rows() -> None:
    frame = pl.DataFrame({"col1": [], "col2": [], "col3": []})
    transformer = Sort(columns=["col3", "col1"])
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_sort_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Sort(columns=["col3", "col1", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 1, None, 2],
                "col2": [None, 6.0, 4.0, 5.0],
                "col3": [None, "a", "b", "c"],
            }
        ),
    )


def test_sort_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Sort(columns=["col3", "col1", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_sort_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = Sort(columns=["col3", "col1", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [None, 1, None, 2],
                "col2": [None, 6.0, 4.0, 5.0],
                "col3": [None, "a", "b", "c"],
            }
        ),
    )


############################################
#     Tests for SortColumnsTransformer     #
############################################


def test_sort_columns_transformer_repr() -> None:
    assert repr(SortColumns()).startswith("SortColumnsTransformer(")


def test_sort_columns_transformer_str() -> None:
    assert str(SortColumns()).startswith("SortColumnsTransformer(")


def test_sort_columns_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = SortColumns()
    frame = pl.DataFrame({"col2": [1, 2, None], "col3": [6.0, 5.0, 4.0], "col1": ["a", "c", "b"]})
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'SortColumnsTransformer.fit' as there are no parameters available to fit"
    )


def test_sort_columns_fit_transformer_transform() -> None:
    frame = pl.DataFrame({"col2": [1, 2, None], "col3": [6.0, 5.0, 4.0], "col1": ["a", "c", "b"]})
    transformer = SortColumns()
    out = transformer.fit_transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col1": ["a", "c", "b"], "col2": [1, 2, None], "col3": [6.0, 5.0, 4.0]}),
    )


def test_sort_columns_transformer_transform() -> None:
    frame = pl.DataFrame({"col2": [1, 2, None], "col3": [6.0, 5.0, 4.0], "col1": ["a", "c", "b"]})
    transformer = SortColumns()
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col1": ["a", "c", "b"], "col2": [1, 2, None], "col3": [6.0, 5.0, 4.0]}),
    )


def test_sort_columns_transformer_transform_reverse() -> None:
    frame = pl.DataFrame({"col2": [1, 2, None], "col3": [6.0, 5.0, 4.0], "col1": ["a", "c", "b"]})
    transformer = SortColumns(reverse=True)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame({"col3": [6.0, 5.0, 4.0], "col2": [1, 2, None], "col1": ["a", "c", "b"]}),
    )
