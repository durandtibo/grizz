# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from grizz.ingestor import BaseIngestor, ParquetIngestor
from grizz.transformer import (
    AbsDiffHorizontal,
    BaseTransformer,
    DiffHorizontal,
    Sequential,
)

logger = logging.getLogger(__name__)


def make_ingestor(data_path: Path) -> BaseIngestor:
    """Define the DataFrame ingestor.

    Args:
        data_path: The path where to store the data for this demo example.
    """
    n_samples = 100
    rng = np.random.default_rng(42)

    data = pl.DataFrame(
        {
            "col1": rng.normal(size=(n_samples,)),
            "col2": rng.uniform(size=(n_samples,)),
        }
    )

    path = data_path.joinpath("data.parquet")
    path.parent.mkdir(exist_ok=True, parents=True)
    data.write_parquet(path)
    return ParquetIngestor(path)


def make_transformer(data_path: Path) -> BaseTransformer:  # noqa: ARG001
    """Define the DataFrame transformer.

    Args:
        data_path: The path where to store the data for this demo example.
    """
    return Sequential(
        [
            DiffHorizontal(in1_col="col1", in2_col="col2", out_col="diff"),
            AbsDiffHorizontal(in1_col="col1", in2_col="col2", out_col="abs_diff"),
        ]
    )


def make_and_run_pipeline(data_path: Path) -> None:
    """Define the pipeline and run it.

    Args:
        data_path: The path where to store the data for this demo example.
    """
    ingestor = make_ingestor(data_path)
    logger.info(f"ingestor:\n{ingestor}")
    data = ingestor.ingest()
    logger.info(f"ingested data: {data}")

    transformer = make_transformer(data_path)
    logger.info(f"transformer:\n{transformer}")
    out = transformer.fit_transform(data)
    logger.info(f"transformed data: {out}")


def main() -> None:
    r"""Define the main function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir).joinpath("data")
        make_and_run_pipeline(data_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # configure_logging(level=logging.INFO)
    main()
