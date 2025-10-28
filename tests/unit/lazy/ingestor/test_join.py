from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.lazy.ingestor import BaseIngestor, Ingestor, JoinIngestor

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def ingestors() -> tuple[BaseIngestor, ...]:
    ingestor1 = Ingestor(
        frame=pl.LazyFrame(
            {
                "col": [1, 2, 3, 4, 5],
                "col1": ["1", "2", "3", "4", "5"],
                "col2": ["a", "b", "c", "d", "e"],
            }
        )
    )
    ingestor2 = Ingestor(
        frame=pl.LazyFrame(
            {
                "col": [1, 2, 3, 5],
                "col3": [-1, -2, -3, -5],
            }
        )
    )
    ingestor3 = Ingestor(
        frame=pl.LazyFrame(
            {
                "col": [1, 2, 3, 4, 5],
                "col4": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col5": ["1.1", "2.2", "3.3", "4.4", "5.5"],
            }
        )
    )
    return (ingestor1, ingestor2, ingestor3)


##################################
#     Tests for JoinIngestor     #
##################################


def test_join_ingestor_incorrect_ingestors() -> None:
    with pytest.raises(ValueError, match=r"'ingestors' must contain at least one ingestor"):
        JoinIngestor([])


def test_join_ingestor_repr(ingestors: Sequence[BaseIngestor]) -> None:
    assert repr(JoinIngestor(ingestors)).startswith("JoinIngestor(")


def test_join_ingestor_str(ingestors: Sequence[BaseIngestor]) -> None:
    assert str(JoinIngestor(ingestors)).startswith("JoinIngestor(")


def test_join_ingestor_equal_true(ingestors: Sequence[BaseIngestor]) -> None:
    assert JoinIngestor(ingestors).equal(JoinIngestor(ingestors))


def test_join_ingestor_equal_false_different_ingestors(ingestors: Sequence[BaseIngestor]) -> None:
    assert not JoinIngestor(ingestors).equal(
        JoinIngestor(
            [
                Ingestor(
                    frame=pl.LazyFrame(
                        {
                            "col": [1, 2, 3, 4, 5],
                            "col1": ["1", "2", "3", "4", "5"],
                            "col2": ["a", "b", "c", "d", "e"],
                        }
                    )
                )
            ]
        )
    )


def test_join_ingestor_equal_false_different_kwargs(ingestors: Sequence[BaseIngestor]) -> None:
    assert not JoinIngestor(ingestors).equal(JoinIngestor(ingestors, on="col"))


def test_join_ingestor_equal_false_different_type(ingestors: Sequence[BaseIngestor]) -> None:
    assert not JoinIngestor(ingestors).equal(42)


def test_join_ingestor_ingest_inner(ingestors: Sequence[BaseIngestor]) -> None:
    assert_frame_equal(
        JoinIngestor(ingestors, on="col", how="inner").ingest(),
        pl.LazyFrame(
            {
                "col": [1, 2, 3, 5],
                "col1": ["1", "2", "3", "5"],
                "col2": ["a", "b", "c", "e"],
                "col3": [-1, -2, -3, -5],
                "col4": [1.1, 2.2, 3.3, 5.5],
                "col5": ["1.1", "2.2", "3.3", "5.5"],
            }
        ),
    )


def test_join_ingestor_ingest_left(ingestors: Sequence[BaseIngestor]) -> None:
    assert_frame_equal(
        JoinIngestor(ingestors, on="col", how="left").ingest(),
        pl.LazyFrame(
            {
                "col": [1, 2, 3, 4, 5],
                "col1": ["1", "2", "3", "4", "5"],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [-1, -2, -3, None, -5],
                "col4": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col5": ["1.1", "2.2", "3.3", "4.4", "5.5"],
            }
        ),
    )
