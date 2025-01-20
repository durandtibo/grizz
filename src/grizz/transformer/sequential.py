r"""Contain a transformer to combine sequentially multiple
transformers."""

from __future__ import annotations

__all__ = ["SequentialTransformer"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import repr_indent, repr_sequence, str_indent, str_sequence

from grizz.transformer.base import BaseTransformer, setup_transformer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


class SequentialTransformer(BaseTransformer):
    r"""Implement a ``polars.DataFrame`` transformer to apply
    sequentially several transformers.

    Args:
        transformers: The transformers or their configurations.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import Sequential, InplaceCast
    >>> transformer = Sequential(
    ...     [
    ...         InplaceCast(columns=["col1"], dtype=pl.Float32),
    ...         InplaceCast(columns=["col2"], dtype=pl.Int64),
    ...     ]
    ... )
    >>> transformer
    SequentialTransformer(
      (0): InplaceCastTransformer(columns=('col1',), exclude_columns=(), missing_policy='raise', dtype=Float32)
      (1): InplaceCastTransformer(columns=('col2',), exclude_columns=(), missing_policy='raise', dtype=Int64)
    )
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬───────┬───────┐
    │ col1 ┆ col2 ┆ col3  ┆ col4  │
    │ ---  ┆ ---  ┆ ---   ┆ ---   │
    │ i64  ┆ str  ┆ str   ┆ str   │
    ╞══════╪══════╪═══════╪═══════╡
    │ 1    ┆ 1    ┆ a     ┆ a     │
    │ 2    ┆ 2    ┆  b    ┆  b    │
    │ 3    ┆ 3    ┆   c   ┆   c   │
    │ 4    ┆ 4    ┆ d     ┆ d     │
    │ 5    ┆ 5    ┆ e     ┆ e     │
    └──────┴──────┴───────┴───────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬───────┬───────┐
    │ col1 ┆ col2 ┆ col3  ┆ col4  │
    │ ---  ┆ ---  ┆ ---   ┆ ---   │
    │ f32  ┆ i64  ┆ str   ┆ str   │
    ╞══════╪══════╪═══════╪═══════╡
    │ 1.0  ┆ 1    ┆ a     ┆ a     │
    │ 2.0  ┆ 2    ┆  b    ┆  b    │
    │ 3.0  ┆ 3    ┆   c   ┆   c   │
    │ 4.0  ┆ 4    ┆ d     ┆ d     │
    │ 5.0  ┆ 5    ┆ e     ┆ e     │
    └──────┴──────┴───────┴───────┘

    ```
    """

    def __init__(self, transformers: Sequence[BaseTransformer | dict]) -> None:
        self._transformers = tuple(setup_transformer(transformer) for transformer in transformers)

    def __repr__(self) -> str:
        args = ""
        if self._transformers:
            args = f"\n  {repr_indent(repr_sequence(self._transformers))}\n"
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = ""
        if self._transformers:
            args = f"\n  {str_indent(str_sequence(self._transformers))}\n"
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._transformers, other._transformers, equal_nan=equal_nan)

    def fit(self, frame: pl.DataFrame) -> None:
        for transformer in self._transformers:
            transformer.fit(frame)

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for transformer in self._transformers:
            frame = transformer.fit_transform(frame)
        return frame

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for transformer in self._transformers:
            frame = transformer.transform(frame)
        return frame
