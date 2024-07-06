from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from grizz.ingestor import CsvIngestor, is_ingestor_config, setup_ingestor

if TYPE_CHECKING:
    import pytest

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
