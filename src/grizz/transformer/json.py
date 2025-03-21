r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = ["InplaceJsonDecodeTransformer", "JsonDecodeTransformer"]

import logging
from typing import TYPE_CHECKING, Any, Union

import polars as pl
import polars.selectors as cs

from grizz.transformer.columns import BaseInNOutNTransformer
from grizz.transformer.utils import get_classname, message_skip_fit

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

PolarsDataType = Union[pl.DataType, type[pl.DataType]]


class JsonDecodeTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to parse string values as JSON.

    Args:
        columns: The columns to parse. ``None`` means all the
            columns.
        prefix: The column name prefix for the output columns.
        suffix: The column name suffix for the output columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
        exist_policy: The policy on how to handle existing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column already exist.
            If ``'warn'``, a warning is raised if at least one column
            already exist and the existing columns are overwritten.
            If ``'ignore'``, the existing columns are overwritten and
            no warning message appears.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message appears.
        **kwargs: Additional arguments passed to
            ``polars.Expr.str.json_decode``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import JsonDecode
    >>> transformer = JsonDecode(columns=["col1", "col3"], prefix="", suffix="_out")
    >>> transformer
    JsonDecodeTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')
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
    shape: (5, 6)
    ┌───────────┬──────┬─────────────────┬──────┬───────────┬─────────────────┐
    │ col1      ┆ col2 ┆ col3            ┆ col4 ┆ col1_out  ┆ col3_out        │
    │ ---       ┆ ---  ┆ ---             ┆ ---  ┆ ---       ┆ ---             │
    │ str       ┆ str  ┆ str             ┆ str  ┆ list[i64] ┆ list[str]       │
    ╞═══════════╪══════╪═════════════════╪══════╪═══════════╪═════════════════╡
    │ [1, 2]    ┆ 1    ┆ ['1', '2']      ┆ a    ┆ [1, 2]    ┆ ["1", "2"]      │
    │ [2]       ┆ 2    ┆ ['2']           ┆ b    ┆ [2]       ┆ ["2"]           │
    │ [1, 2, 3] ┆ 3    ┆ ['1', '2', '3'] ┆ c    ┆ [1, 2, 3] ┆ ["1", "2", "3"] │
    │ [4, 5]    ┆ 4    ┆ ['4', '5']      ┆ d    ┆ [4, 5]    ┆ ["4", "5"]      │
    │ [5, 4]    ┆ 5    ┆ ['5', '4']      ┆ e    ┆ [5, 4]    ┆ ["5", "4"]      │
    └───────────┴──────┴─────────────────┴──────┴───────────┴─────────────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix=prefix,
            suffix=suffix,
            exclude_columns=exclude_columns,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        columns = self.find_common_columns(frame)
        logger.info(f"Converting {len(columns):,} columns to JSON...")
        return frame.select(
            (cs.by_name(columns) & cs.string())
            .str.replace_all("'", '"')
            .str.json_decode(**self._kwargs)
        )


class InplaceJsonDecodeTransformer(JsonDecodeTransformer):
    r"""Implement a transformer to parse string values as JSON.

    Args:
        columns: The columns to parse. ``None`` means all the
            columns.
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
        **kwargs: Additional arguments passed to
            ``polars.Expr.str.json_decode``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import InplaceJsonDecode
    >>> transformer = InplaceJsonDecode(columns=["col1", "col3"])
    >>> transformer
    InplaceJsonDecodeTransformer(columns=('col1', 'col3'), exclude_columns=(), missing_policy='raise')
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
        columns: Sequence[str] | None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            prefix="",
            suffix="",
            exclude_columns=exclude_columns,
            exist_policy="ignore",
            missing_policy=missing_policy,
            **kwargs,
        )

    def get_args(self) -> dict:
        args = super().get_args()
        for key in ["prefix", "suffix", "exist_policy"]:
            args.pop(key)
        return args
