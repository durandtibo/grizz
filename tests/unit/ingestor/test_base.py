from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester
from objectory import OBJECT_TARGET

from grizz.ingestor import (
    CsvIngestor,
    ParquetIngestor,
    is_ingestor_config,
    setup_ingestor,
)
from grizz.ingestor.base import IngestorEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


########################################
#     Tests for is_ingestor_config     #
########################################


def test_is_ingestor_config_true() -> None:
    assert is_ingestor_config(
        {OBJECT_TARGET: "grizz.ingestor.CsvIngestor", "path": "/path/to/data.csv"}
    )


def test_is_ingestor_config_false() -> None:
    assert not is_ingestor_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_ingestor     #
####################################


def test_setup_ingestor_object() -> None:
    ingestor = CsvIngestor(path="/path/to/data.csv")
    assert setup_ingestor(ingestor) is ingestor


def test_setup_ingestor_dict() -> None:
    assert isinstance(
        setup_ingestor({OBJECT_TARGET: "grizz.ingestor.CsvIngestor", "path": "/path/to/data.csv"}),
        CsvIngestor,
    )


def test_setup_ingestor_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_ingestor({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages


################################################
#     Tests for IngestorEqualityComparator     #
################################################


INGESTOR_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=CsvIngestor("data.csv"),
            expected=CsvIngestor("data.csv"),
        ),
        id="CSV ingestor",
    ),
    pytest.param(
        ExamplePair(
            actual=ParquetIngestor("data.parquet"),
            expected=ParquetIngestor("data.parquet"),
        ),
        id="parquet ingestor",
    ),
]


INGESTOR_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=42.0,
            expected=CsvIngestor("data.csv"),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=CsvIngestor("data.csv"),
            expected=CsvIngestor("data2.csv"),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_ingestor_equality_comparator_repr() -> None:
    assert repr(IngestorEqualityComparator()) == "IngestorEqualityComparator()"


def test_ingestor_equality_comparator_str() -> None:
    assert str(IngestorEqualityComparator()) == "IngestorEqualityComparator()"


def test_ingestor_equality_comparator__eq__true() -> None:
    assert IngestorEqualityComparator() == IngestorEqualityComparator()


def test_ingestor_equality_comparator__eq__false() -> None:
    assert IngestorEqualityComparator() != 123


def test_ingestor_equality_comparator_clone() -> None:
    op = IngestorEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_ingestor_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = CsvIngestor("data.csv")
    assert IngestorEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", INGESTOR_EQUAL)
def test_ingestor_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = IngestorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", INGESTOR_EQUAL)
def test_ingestor_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = IngestorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", INGESTOR_NOT_EQUAL)
def test_ingestor_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = IngestorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", INGESTOR_NOT_EQUAL)
def test_ingestor_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = IngestorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", INGESTOR_EQUAL)
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
@pytest.mark.parametrize("example", INGESTOR_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", INGESTOR_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
