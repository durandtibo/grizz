from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from grizz.exporter import CsvExporter, is_exporter_config, setup_exporter

if TYPE_CHECKING:
    import pytest

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
