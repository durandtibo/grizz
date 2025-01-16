r"""Contain the implementation of parquet ingestors."""

from __future__ import annotations

__all__ = ["JoinIngestor"]

import logging
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import repr_indent, repr_sequence
from coola.utils.format import repr_mapping, repr_mapping_line
from iden.utils.time import timeblock

from grizz.ingestor import setup_ingestor
from grizz.ingestor.base import BaseIngestor
from grizz.utils.format import human_byte

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

logger = logging.getLogger(__name__)


class JoinIngestor(BaseIngestor):
    r"""Implement an ingestor that joins the output of multiple
    ingestors.

    Args:
        ingestors: The list of ingestors.
        **kwargs: Additional keyword arguments for
            ``polars.DataFrame.join``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.ingestor import JoinIngestor, Ingestor
    >>> ingestor1 = Ingestor(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col": [1, 2, 3, 4, 5],
    ...             "col1": ["1", "2", "3", "4", "5"],
    ...             "col2": ["a", "b", "c", "d", "e"],
    ...         }
    ...     )
    ... )
    >>> ingestor2 = Ingestor(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col": [1, 2, 3, 5],
    ...             "col3": [-1, -2, -3, -5],
    ...         }
    ...     )
    ... )
    >>> ingestor3 = Ingestor(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col": [1, 2, 3, 4, 5],
    ...             "col4": [1.1, 2.2, 3.3, 4.4, 5.5],
    ...             "col5": ["1.1", "2.2", "3.3", "4.4", "5.5"],
    ...         }
    ...     )
    ... )
    >>> ingestor = JoinIngestor([ingestor1, ingestor2, ingestor3], on="col", how="inner")
    >>> ingestor
    JoinIngestor(
      (ingestors):
        (0): Ingestor(shape=(5, 3))
        (1): Ingestor(shape=(4, 2))
        (2): Ingestor(shape=(5, 3))
      (kwargs): on='col', how='inner'
    )
    >>> frame = ingestor.ingest()
    >>> frame
    shape: (4, 6)
    ┌─────┬──────┬──────┬──────┬──────┬──────┐
    │ col ┆ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col5 │
    │ --- ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64 ┆ str  ┆ str  ┆ i64  ┆ f64  ┆ str  │
    ╞═════╪══════╪══════╪══════╪══════╪══════╡
    │ 1   ┆ 1    ┆ a    ┆ -1   ┆ 1.1  ┆ 1.1  │
    │ 2   ┆ 2    ┆ b    ┆ -2   ┆ 2.2  ┆ 2.2  │
    │ 3   ┆ 3    ┆ c    ┆ -3   ┆ 3.3  ┆ 3.3  │
    │ 5   ┆ 5    ┆ e    ┆ -5   ┆ 5.5  ┆ 5.5  │
    └─────┴──────┴──────┴──────┴──────┴──────┘

    ```
    """

    def __init__(self, ingestors: Sequence[BaseIngestor | dict], **kwargs: Any) -> None:
        if len(ingestors) < 1:
            msg = "'ingestors' must contain at least one ingestor"
            raise ValueError(msg)
        self._ingestors = tuple(setup_ingestor(ingestor) for ingestor in ingestors)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "ingestors": "\n" + repr_sequence(self._ingestors),
                    "kwargs": repr_mapping_line(self._kwargs),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self._ingestors, other._ingestors, equal_nan=equal_nan
        ) and objects_are_equal(self._kwargs, other._kwargs, equal_nan=equal_nan)

    def ingest(self) -> pl.DataFrame:
        logger.info("Joining DataFrames...")
        with timeblock("join time: {time}"):
            frame = self._ingest()
            logger.info(
                f"DataFrame ingested | shape={frame.shape}  "
                f"estimated size={human_byte(frame.estimated_size())}"
            )
        return frame

    def _ingest(self) -> pl.DataFrame:
        out = self._ingestors[0].ingest()
        for ingestor in self._ingestors[1:]:
            frame = ingestor.ingest()
            out = out.join(frame, **self._kwargs)
        return out
