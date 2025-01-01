from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester
from objectory import OBJECT_TARGET

from grizz.exporter import (
    CsvExporter,
    ParquetExporter,
    is_exporter_config,
    setup_exporter,
)
from grizz.exporter.base import ExporterEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


########################################
#     Tests for is_exporter_config     #
########################################


def test_is_exporter_config_true() -> None:
    assert is_exporter_config(
        {OBJECT_TARGET: "grizz.exporter.CsvExporter", "path": "/path/to/data.csv"}
    )


def test_is_exporter_config_false() -> None:
    assert not is_exporter_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_exporter     #
####################################


def test_setup_exporter_object() -> None:
    exporter = CsvExporter(path="/path/to/data.csv")
    assert setup_exporter(exporter) is exporter


def test_setup_exporter_dict() -> None:
    assert isinstance(
        setup_exporter({OBJECT_TARGET: "grizz.exporter.CsvExporter", "path": "/path/to/data.csv"}),
        CsvExporter,
    )


def test_setup_exporter_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_exporter({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages


################################################
#     Tests for ExporterEqualityComparator     #
################################################


EXPORTER_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=CsvExporter("data.csv"),
            expected=CsvExporter("data.csv"),
        ),
        id="CSV exporter",
    ),
    pytest.param(
        ExamplePair(
            actual=ParquetExporter("data.parquet"),
            expected=ParquetExporter("data.parquet"),
        ),
        id="parquet exporter",
    ),
]


EXPORTER_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=42.0,
            expected=CsvExporter("data.csv"),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=CsvExporter("data.csv"),
            expected=CsvExporter("data2.csv"),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_exporter_equality_comparator_repr() -> None:
    assert repr(ExporterEqualityComparator()) == "ExporterEqualityComparator()"


def test_exporter_equality_comparator_str() -> None:
    assert str(ExporterEqualityComparator()) == "ExporterEqualityComparator()"


def test_exporter_equality_comparator__eq__true() -> None:
    assert ExporterEqualityComparator() == ExporterEqualityComparator()


def test_exporter_equality_comparator__eq__false() -> None:
    assert ExporterEqualityComparator() != 123


def test_exporter_equality_comparator_clone() -> None:
    op = ExporterEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_exporter_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = CsvExporter("data.csv")
    assert ExporterEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", EXPORTER_EQUAL)
def test_exporter_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EXPORTER_EQUAL)
def test_exporter_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EXPORTER_NOT_EQUAL)
def test_exporter_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EXPORTER_NOT_EQUAL)
def test_exporter_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", EXPORTER_EQUAL)
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
@pytest.mark.parametrize("example", EXPORTER_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", EXPORTER_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
