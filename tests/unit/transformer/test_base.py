from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl
import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester
from objectory import OBJECT_TARGET

from grizz.transformer import (
    Cast,
    FillNan,
    FillNull,
    is_transformer_config,
    setup_transformer,
)
from grizz.transformer.base import TransformerEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


###########################################
#     Tests for is_transformer_config     #
###########################################


def test_is_transformer_config_true() -> None:
    assert is_transformer_config(
        {
            OBJECT_TARGET: "grizz.transformer.Cast",
            "columns": ["col1", "col3"],
            "dtype": pl.Int32,
        }
    )


def test_is_transformer_config_false() -> None:
    assert not is_transformer_config({OBJECT_TARGET: "collections.Counter"})


#######################################
#     Tests for setup_transformer     #
#######################################


def test_setup_transformer_object() -> None:
    transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    assert setup_transformer(transformer) is transformer


def test_setup_transformer_dict() -> None:
    assert isinstance(
        setup_transformer(
            {
                OBJECT_TARGET: "grizz.transformer.Cast",
                "columns": ["col1", "col3"],
                "dtype": pl.Int32,
            }
        ),
        Cast,
    )


def test_setup_transformer_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_transformer({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages


###################################################
#     Tests for TransformerEqualityComparator     #
###################################################


TRANSFORMER_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=FillNan(columns=["col1", "col4"]),
            expected=FillNan(columns=["col1", "col4"]),
        ),
        id="CSV transformer",
    ),
    pytest.param(
        ExamplePair(
            actual=FillNull(columns=["col1", "col4"]),
            expected=FillNull(columns=["col1", "col4"]),
        ),
        id="parquet transformer",
    ),
]


TRANSFORMER_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=42.0,
            expected=FillNan(columns=["col1", "col4"]),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=FillNan(columns=["col1", "col4"]),
            expected=FillNan(columns=["col1", "col2", "col4"]),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_transformer_equality_comparator_repr() -> None:
    assert repr(TransformerEqualityComparator()) == "TransformerEqualityComparator()"


def test_transformer_equality_comparator_str() -> None:
    assert str(TransformerEqualityComparator()) == "TransformerEqualityComparator()"


def test_transformer_equality_comparator__eq__true() -> None:
    assert TransformerEqualityComparator() == TransformerEqualityComparator()


def test_transformer_equality_comparator__eq__false() -> None:
    assert TransformerEqualityComparator() != 123


def test_transformer_equality_comparator_clone() -> None:
    op = TransformerEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_transformer_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = FillNan(columns=["col1", "col4"])
    assert TransformerEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", TRANSFORMER_EQUAL)
def test_transformer_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = TransformerEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", TRANSFORMER_EQUAL)
def test_transformer_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = TransformerEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", TRANSFORMER_NOT_EQUAL)
def test_transformer_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = TransformerEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", TRANSFORMER_NOT_EQUAL)
def test_transformer_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = TransformerEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", TRANSFORMER_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert function(example.actual, example.expected, show_difference=show_difference)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", TRANSFORMER_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", TRANSFORMER_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
