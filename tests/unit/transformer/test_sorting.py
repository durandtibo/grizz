from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
from polars.testing import assert_frame_equal

from grizz.transformer import Sort, SortColumns

if TYPE_CHECKING:
    import pytest

#####################################
#     Tests for SortTransformer     #
#####################################


def test_sort_transformer_repr() -> None:
    assert repr(Sort(columns=["col3", "col1"])).startswith("SortTransformer(")


def test_sort_transformer_str() -> None:
    assert str(Sort(columns=["col3", "col1"])).startswith("SortTransformer(")


def test_sort_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = Sort(columns=["col3", "col1"])
    frame = pl.DataFrame(
        {"col1": [None, 1, 2, None], "col2": [None, 6.0, 5.0, 4.0], "col3": [None, "a", "c", "b"]}
    )
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'SortTransformer.fit' as there are no parameters available to fit"
    )


def test_sort_transformer_fit_transform() -> None:
    frame = pl.DataFrame(
        {"col1": [None, 1, 2, None], "col2": [None, 6.0, 5.0, 4.0], "col3": [None, "a", "c", "b"]}
    )
    transformer = Sort(columns=["col3", "col1"])
    out = transformer.fit_transform(frame)
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


def test_sort_transformer_transform() -> None:
    frame = pl.DataFrame(
        {"col1": [None, 1, 2, None], "col2": [None, 6.0, 5.0, 4.0], "col3": [None, "a", "c", "b"]}
    )
    transformer = Sort(columns=["col3", "col1"])
    out = transformer.transform(frame)
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


def test_sort_transformer_transform_null_last() -> None:
    frame = pl.DataFrame(
        {"col1": [None, 1, 2, None], "col2": [None, 6.0, 5.0, 4.0], "col3": [None, "a", "c", "b"]}
    )
    transformer = Sort(columns=["col3", "col1"], nulls_last=True)
    out = transformer.transform(frame)
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


def test_sort_transformer_transform_empty() -> None:
    frame = pl.DataFrame({"col1": [], "col2": [], "col3": []})
    transformer = Sort(columns=["col3", "col1"])
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col1": [], "col2": [], "col3": []}))


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
