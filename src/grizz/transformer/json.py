r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = ["JsonDecodeTransformer"]

import logging
from typing import TYPE_CHECKING, Any, Union

import polars as pl
import polars.selectors as cs
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_kwargs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars.type_aliases import PythonDataType

logger = logging.getLogger(__name__)


PolarsDataType = Union[pl.DataType, type[pl.DataType]]


class JsonDecodeTransformer(BaseInNTransformer):
    r"""Implement a transformer to parse string values as JSON.

    Args:
        columns: The columns to parse. ``None`` means all the
            columns.
        dtype: The dtype to cast the extracted value to.
            If ``None``, the dtype will be inferred from the JSON
            value.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message appears.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import JsonDecode
    >>> transformer = JsonDecode(columns=["col1", "col3"])
    >>> transformer
    JsonDecodeTransformer(columns=('col1', 'col3'), dtype=None, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["[1, 2]", "[2]", "[1, 2, 3]", "[4, 5]", "[5, 4]"],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["['1', '2']", "['2']", "['1', '2', '3']", "['4', '5']", "['5', '4']"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌───────────┬──────┬─────────────────┬──────┐
    │ col1      ┆ col2 ┆ col3            ┆ col4 │
    │ ---       ┆ ---  ┆ ---             ┆ ---  │
    │ str       ┆ str  ┆ str             ┆ str  │
    ╞═══════════╪══════╪═════════════════╪══════╡
    │ [1, 2]    ┆ 1    ┆ ['1', '2']      ┆ a    │
    │ [2]       ┆ 2    ┆ ['2']           ┆ b    │
    │ [1, 2, 3] ┆ 3    ┆ ['1', '2', '3'] ┆ c    │
    │ [4, 5]    ┆ 4    ┆ ['4', '5']      ┆ d    │
    │ [5, 4]    ┆ 5    ┆ ['5', '4']      ┆ e    │
    └───────────┴──────┴─────────────────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌───────────┬──────┬─────────────────┬──────┐
    │ col1      ┆ col2 ┆ col3            ┆ col4 │
    │ ---       ┆ ---  ┆ ---             ┆ ---  │
    │ list[i64] ┆ str  ┆ list[str]       ┆ str  │
    ╞═══════════╪══════╪═════════════════╪══════╡
    │ [1, 2]    ┆ 1    ┆ ["1", "2"]      ┆ a    │
    │ [2]       ┆ 2    ┆ ["2"]           ┆ b    │
    │ [1, 2, 3] ┆ 3    ┆ ["1", "2", "3"] ┆ c    │
    │ [4, 5]    ┆ 4    ┆ ["4", "5"]      ┆ d    │
    │ [5, 4]    ┆ 5    ┆ ["5", "4"]      ┆ e    │
    └───────────┴──────┴─────────────────┴──────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        dtype: PolarsDataType | PythonDataType | None = None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._dtype = dtype
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "dtype": self._dtype,
                "exclude_columns": self._exclude_columns,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args}{str_kwargs(self._kwargs)})"

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Converting {len(self.find_columns(frame)):,} columns to JSON...")
        columns = self.find_common_columns(frame)
        return frame.with_columns(
            frame.select(
                (cs.by_name(columns) & cs.string())
                .str.replace_all("'", '"')
                .str.json_decode(self._dtype, **self._kwargs)
            )
        )
