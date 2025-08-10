from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import TransformerNotFittedError
from grizz.transformer.abc import BaseArgTransformer


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": ["2020-1-1", None, "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, None],
            "col3": [None, None, None, None, None],
        }
    )


########################################
#     Tests for BaseArgTransformer     #
########################################


class MyArgTransformer(BaseArgTransformer):
    def __init__(self, col: str, value: Any, requires_fit: bool = False) -> None:
        super().__init__()
        self._col = col
        self._value = value

        self._requires_fit = requires_fit

    def check_is_fitted(self) -> None:
        if self._requires_fit and not self._is_fitted:
            msg = "This transformer instance is not fitted yet"
            raise TransformerNotFittedError(msg)

    def get_args(self) -> dict:
        return {"col": self._col, "value": self._value}

    def _fit(self, frame: pl.DataFrame) -> None:
        pass

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col(self._col).fill_null(self._value))


def test_base_arg_transformer_repr() -> None:
    assert repr(MyArgTransformer(col="col", value=-1)) == "MyArgTransformer(col='col', value=-1)"


def test_base_arg_transformer_str() -> None:
    assert str(MyArgTransformer(col="col", value=-1)) == "MyArgTransformer(col=col, value=-1)"


def test_base_arg_transformer_equal_true() -> None:
    assert MyArgTransformer(col="col", value=-1).equal(MyArgTransformer(col="col", value=-1))


def test_base_arg_transformer_equal_false_different_col() -> None:
    assert not MyArgTransformer(col="col", value=-1).equal(MyArgTransformer(col="col1", value=-1))


def test_base_arg_transformer_equal_false_different_value() -> None:
    assert not MyArgTransformer(col="col", value=-1).equal(MyArgTransformer(col="col", value=0))


def test_base_arg_transformer_equal_false_different_type() -> None:
    assert not MyArgTransformer(col="col", value=-1).equal(42)


def test_base_arg_transformer_get_args() -> None:
    assert objects_are_equal(
        MyArgTransformer(col="col", value=-1).get_args(), {"col": "col", "value": -1}
    )


def test_base_arg_transformer_check_is_fitted_requires_fit_false() -> None:
    MyArgTransformer(col="col2", value=-1).check_is_fitted()


def test_base_arg_transformer_check_is_fitted_requires_fit_true() -> None:
    transformer = MyArgTransformer(col="col2", value=-1, requires_fit=True)
    with pytest.raises(
        TransformerNotFittedError, match="This transformer instance is not fitted yet"
    ):
        transformer.check_is_fitted()


def test_base_arg_transformer_fit(dataframe: pl.DataFrame) -> None:
    MyArgTransformer(col="col", value=-1).fit(dataframe)


def test_base_arg_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = MyArgTransformer(col="col2", value=-1)
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
    transformer = MyArgTransformer(col="col2", value=-1)
    out = transformer.transform(dataframe)
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


def test_base_arg_transformer_transform_requires_fit(dataframe: pl.DataFrame) -> None:
    transformer = MyArgTransformer(col="col2", value=-1, requires_fit=True)
    with pytest.raises(
        TransformerNotFittedError, match="This transformer instance is not fitted yet"
    ):
        transformer.transform(dataframe)
