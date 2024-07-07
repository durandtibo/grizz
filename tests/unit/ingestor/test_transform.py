from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.ingestor import ParquetIngestor, TransformIngestor
from grizz.transformer import Cast

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("frame.parquet")
    frame = pl.DataFrame(
        {
            "col1": ["1", "2", "3", "4", "5"],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    frame.write_parquet(path)
    return path


#######################################
#     Tests for TransformIngestor     #
#######################################


def test_transform_ingestor_repr(frame_path: Path) -> None:
    assert repr(
        TransformIngestor(
            ingestor=ParquetIngestor(path=frame_path),
            transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
        )
    ).startswith("TransformIngestor(")


def test_transform_ingestor_str(frame_path: Path) -> None:
    assert str(
        TransformIngestor(
            ingestor=ParquetIngestor(path=frame_path),
            transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
        )
    ).startswith("TransformIngestor(")


def test_transform_ingestor_ingest(frame_path: Path) -> None:
    ingestor = TransformIngestor(
        ingestor=ParquetIngestor(path=frame_path),
        transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
    )
    assert_frame_equal(
        ingestor.ingest(),
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            },
            schema={"col1": pl.Float32, "col2": pl.String, "col3": pl.Float32},
        ),
    )
