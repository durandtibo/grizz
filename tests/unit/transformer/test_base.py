from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl
from objectory import OBJECT_TARGET

from grizz.transformer import Cast, is_transformer_config, setup_transformer

if TYPE_CHECKING:
    import pytest

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
