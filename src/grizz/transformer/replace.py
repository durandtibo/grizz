r"""Contain transformers to replace values."""

from __future__ import annotations

__all__ = ["ReplaceStrictTransformer", "ReplaceTransformer"]

import logging
from typing import Any

import polars as pl
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseIn1Out1Transformer
from grizz.utils.format import str_kwargs

logger = logging.getLogger(__name__)


class ReplaceTransformer(BaseIn1Out1Transformer):
    r"""Replace the values in a column by the values in a mapping.

    Args:
        in_col: The input column name.
        out_col: The output column name.
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
        **kwargs: The keyword arguments to pass to ``replace``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import Replace
    >>> transformer = Replace(in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3})
    >>> transformer
    ReplaceTransformer(in_col='old', out_col='new', exist_policy='raise', missing_policy='raise', old={'a': 1, 'b': 2, 'c': 3})
    >>> frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    >>> frame
    shape: (5, 1)
    ┌─────┐
    │ old │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ c   │
    │ d   │
    │ e   │
    └─────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬─────┐
    │ old ┆ new │
    │ --- ┆ --- │
    │ str ┆ str │
    ╞═════╪═════╡
    │ a   ┆ 1   │
    │ b   ┆ 2   │
    │ c   ┆ 3   │
    │ d   ┆ d   │
    │ e   ┆ e   │
    └─────┴─────┘
    >>> transformer = Replace(
    ...     in_col="old",
    ...     out_col="new",
    ...     old={"a": 1, "b": 2, "c": 3},
    ...     default=None,
    ... )
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬──────┐
    │ old ┆ new  │
    │ --- ┆ ---  │
    │ str ┆ i64  │
    ╞═════╪══════╡
    │ a   ┆ 1    │
    │ b   ┆ 2    │
    │ c   ┆ 3    │
    │ d   ┆ null │
    │ e   ┆ null │
    └─────┴──────┘

    ```
    """

    def __init__(
        self,
        in_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_col=in_col,
            out_col=out_col,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "in_col": self._in_col,
                "out_col": self._out_col,
                "exist_policy": self._exist_policy,
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
        logger.info(
            f"Replacing values from column {self._in_col!r} and "
            f"saving output in {self._out_col!r} ..."
        )
        return frame.with_columns(pl.col(self._in_col).replace(**self._kwargs).alias(self._out_col))


class ReplaceStrictTransformer(BaseIn1Out1Transformer):
    r"""Replace the values in a column by the values in a mapping.

    Args:
        in_col: The input column name.
        out_col: The output column name.
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
        **kwargs: The keyword arguments to pass to ``replace``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ReplaceStrict
    >>> transformer = ReplaceStrict(
    ...     in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    ... )
    >>> transformer
    ReplaceStrictTransformer(in_col='old', out_col='new', exist_policy='raise', missing_policy='raise', old={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
    >>> frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    >>> frame
    shape: (5, 1)
    ┌─────┐
    │ old │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ c   │
    │ d   │
    │ e   │
    └─────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬─────┐
    │ old ┆ new │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ a   ┆ 1   │
    │ b   ┆ 2   │
    │ c   ┆ 3   │
    │ d   ┆ 4   │
    │ e   ┆ 5   │
    └─────┴─────┘
    >>> transformer = ReplaceStrict(
    ...     in_col="old",
    ...     out_col="new",
    ...     old={"a": 1, "b": 2, "c": 3},
    ...     default=None,
    ... )
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌─────┬──────┐
    │ old ┆ new  │
    │ --- ┆ ---  │
    │ str ┆ i64  │
    ╞═════╪══════╡
    │ a   ┆ 1    │
    │ b   ┆ 2    │
    │ c   ┆ 3    │
    │ d   ┆ null │
    │ e   ┆ null │
    └─────┴──────┘

    ```
    """

    def __init__(
        self,
        in_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_col=in_col,
            out_col=out_col,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "in_col": self._in_col,
                "out_col": self._out_col,
                "exist_policy": self._exist_policy,
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
        logger.info(
            f"Replacing values from column {self._in_col!r} and "
            f"saving output in {self._out_col!r} ..."
        )
        return frame.with_columns(
            pl.col(self._in_col).replace_strict(**self._kwargs).alias(self._out_col)
        )
