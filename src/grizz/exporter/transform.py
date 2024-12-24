r"""Contain a wrapper around an exporter to transform the DataFrame
before to export it."""

from __future__ import annotations

__all__ = ["TransformExporter"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from grizz.exporter.base import BaseExporter, setup_exporter
from grizz.transformer.base import BaseTransformer, setup_transformer

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class TransformExporter(BaseExporter):
    r"""Implement an exporter that transforms the DataFrame before to
    export it.

    Args:
        transformer: The ``polars.DataFrame`` transformer or
            its configuration.
        exporter: The DataFrame exporter or its configuration.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.exporter import TransformExporter, ParquetExporter
    >>> from grizz.transformer import Cast
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> exporter = TransformExporter(
    ...     transformer=Cast(columns=["col1", "col3"], dtype=pl.Float32),
    ...     exporter=ParquetExporter(path="/path/to/frame.parquet"),
    ... )
    >>> exporter
    TransformExporter(
      (transformer): CastTransformer(columns=('col1', 'col3'), dtype=Float32, exclude_columns=(), missing_policy='raise')
      (exporter): ParquetExporter(path=/path/to/frame.parquet)
    )
    >>> exporter.export(frame)  # doctest: +SKIP


    ```
    """

    def __init__(
        self,
        transformer: BaseTransformer | dict,
        exporter: BaseExporter | dict,
    ) -> None:
        self._transformer = setup_transformer(transformer)
        self._exporter = setup_exporter(exporter)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping({"transformer": self._transformer, "exporter": self._exporter})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def export(self, frame: pl.DataFrame) -> None:
        frame = self._transformer.transform(frame)
        self._exporter.export(frame)
