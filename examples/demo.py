# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import polars as pl
from coola.utils.imports import is_numpy_available, numpy_available

from grizz.ingestor import BaseIngestor, ParquetIngestor
from grizz.transformer import (
    AbsDiffHorizontal,
    BaseTransformer,
    ColumnClose,
    CopyColumn,
    Diff,
    DiffHorizontal,
    FloatCast,
    MaxAbsScaler,
    Sequential,
    SumHorizontal,
)
from grizz.utils.imports import is_sklearn_available
from grizz.utils.logging import configure_logging

if is_numpy_available():
    import numpy as np

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
    transformers = [
        CopyColumn(in_col="col1", out_col="cc1"),
        FloatCast(columns=["cc1"], dtype=pl.Float32),
        DiffHorizontal(in1_col="col1", in2_col="col2", out_col="col_diff"),
        AbsDiffHorizontal(in1_col="col1", in2_col="col2", out_col="abs_diff"),
        ColumnClose(actual="col1", expected="cc1", out_col="close_1"),
        Diff(in_col="col1", out_col="diff_1", shift=1),
        SumHorizontal(columns=["col1", "col2"], out_col="sum_12"),
    ]
    if is_sklearn_available():
        transformers.append(MaxAbsScaler(columns=["col1", "col2"], prefix="", suffix="_scaled"))
    return Sequential(transformers)


def make_and_run_pipeline(data_path: Path) -> None:
    """Define the pipeline and run it.

    Args:
        data_path: The path where to store the data for this demo example.
    """
    ingestor = make_ingestor(data_path)
    logger.info(f"ingestor:\n{ingestor}")
    data = ingestor.ingest()
    logger.info(f"ingested data:\n{data}")

    transformer = make_transformer(data_path)
    logger.info(f"transformer:\n{transformer}")
    out = transformer.fit_transform(data)
    logger.info(f"transformed data:\n{out}")


@numpy_available
def main() -> None:
    r"""Define the main function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir).joinpath("data")
        make_and_run_pipeline(data_path)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
