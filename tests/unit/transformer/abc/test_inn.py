from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.transformer.abc import BaseInNTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, None],
            "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
            "col3": ["a", "b", "c", "d", None],
            "col4": [1.2, float("nan"), 3.2, None, 5.2],
        },
        schema={"col1": pl.Int64, "col2": pl.Float64, "col3": pl.String, "col4": pl.Float64},
    )


########################################
#     Tests for BaseInNTransformer     #
########################################


class MyInNTransformer(BaseInNTransformer):

    def __init__(
        self,
        columns: Sequence[str] | None,
        value: Any,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(
            columns=columns, exclude_columns=exclude_columns, missing_policy=missing_policy
        )
        self._value = value

    def get_args(self) -> dict:
        return super().get_args() | {"value": self._value}

    def _fit_data(self, frame: pl.DataFrame) -> None:
        pass

    def _transform_data(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col(self._columns).fill_null(self._value))


def test_base_arg_transformer_repr() -> None:
    assert (
        repr(MyInNTransformer(columns=["col1", "col4"], value=-1))
        == "MyInNTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "missing_policy='raise', value=-1)"
    )


def test_base_arg_transformer_str() -> None:
    assert (
        str(MyInNTransformer(columns=["col1", "col4"], value=-1))
        == "MyInNTransformer(columns=('col1', 'col4'), exclude_columns=(), "
        "missing_policy=raise, value=-1)"
    )


def test_base_arg_transformer_equal_true() -> None:
    assert MyInNTransformer(columns=["col1", "col4"], value=-1).equal(
        MyInNTransformer(columns=["col1", "col4"], value=-1)
    )


def test_base_arg_transformer_equal_false_different_columns() -> None:
    assert not MyInNTransformer(columns=["col1", "col4"], value=-1).equal(
        MyInNTransformer(columns=["col1", "col3", "col4"], value=-1)
    )


def test_base_arg_transformer_equal_false_different_exclude_columns() -> None:
    assert not MyInNTransformer(columns=["col1", "col4"], value=-1).equal(
        MyInNTransformer(columns=["col1", "col4"], value=-1, exclude_columns=["col3"])
    )


def test_base_arg_transformer_equal_false_different_missing_policy() -> None:
    assert not MyInNTransformer(columns=["col1", "col4"], value=-1).equal(
        MyInNTransformer(columns=["col1", "col4"], value=-1, missing_policy="ignore")
    )


def test_base_arg_transformer_equal_false_different_value() -> None:
    assert not MyInNTransformer(columns=["col1", "col4"], value=-1).equal(
        MyInNTransformer(columns=["col1", "col4"], value=0)
    )


def test_base_arg_transformer_equal_false_different_type() -> None:
    assert not MyInNTransformer(columns=["col1", "col4"], value=-1).equal(42)


def test_base_arg_transformer_get_args() -> None:
    assert objects_are_equal(
        MyInNTransformer(columns=["col1", "col4"], value=-1).get_args(),
        {
            "columns": ("col1", "col4"),
            "exclude_columns": (),
            "missing_policy": "raise",
            "value": -1,
        },
    )


def test_base_arg_transformer_fit(dataframe: pl.DataFrame) -> None:
    transformer = MyInNTransformer(columns=["col1", "col4"], value=-1)
    transformer.fit(dataframe)
    assert transformer.get_input_columns() == ("col1", "col4")


def test_base_arg_transformer_fit_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = MyInNTransformer(columns=None, value=-1)
    transformer.fit(dataframe)
    assert transformer.get_input_columns() == ("col1", "col2", "col3", "col4")


def test_base_arg_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = MyInNTransformer(columns=["col1", "col4"], value=-1)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", None, "2020-1-31", "2020-12-31", None],
                "col2": [1, -1, 3, -1, -1],
                "col3": [None, None, None, None, None],
            }
        ),
    )


def test_base_arg_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = MyInNTransformer(columns=["col1", "col4"], value=-1)
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["2020-1-1", None, "2020-1-31", "2020-12-31", None],
                "col2": [1, -1, 3, -1, -1],
                "col3": [None, None, None, None, None],
            }
        ),
    )
