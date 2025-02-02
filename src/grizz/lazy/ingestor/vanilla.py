r"""Contain the implementation of a simple ingestor."""

from __future__ import annotations

__all__ = ["Ingestor"]


from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from grizz.lazy.ingestor.base import BaseIngestor

if TYPE_CHECKING:
    import polars as pl


class Ingestor(BaseIngestor):
    r"""Implement a simple LazyFrame ingestor.

    Args:
        frame: The LazyFrame to ingest.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.lazy.ingestor import Ingestor
    >>> ingestor = Ingestor(
    ...     frame=pl.LazyFrame(
    ...         {
    ...             "col1": [1, 2, 3, 4, 5],
    ...             "col2": ["1", "2", "3", "4", "5"],
    ...             "col3": ["1", "2", "3", "4", "5"],
    ...             "col4": ["a", "b", "c", "d", "e"],
    ...         }
    ...     )
    ... )
    >>> ingestor
    Ingestor(
      (schema): Schema({'col1': Int64, 'col2': String, 'col3': String, 'col4': String})
    )
    >>> frame = ingestor.ingest()
    >>> frame
    <LazyFrame at 0x...>

    ```
    """

    def __init__(self, frame: pl.LazyFrame) -> None:
        self._frame = frame

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"schema": self._frame.collect_schema()}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"schema": self._frame.collect_schema()}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self._frame.collect_schema(), other._frame.collect_schema(), equal_nan=equal_nan
        )

    def ingest(self) -> pl.LazyFrame:
        return self._frame
