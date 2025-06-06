r"""Contain transformers to replace values."""

from __future__ import annotations

__all__ = [
    "InplaceReplaceStrictTransformer",
    "InplaceReplaceTransformer",
    "ReplaceStrictTransformer",
    "ReplaceTransformer",
]

import logging
from typing import Any

import polars as pl

from grizz.transformer.columns import BaseIn1Out1Transformer
from grizz.transformer.utils import get_classname, message_skip_fit

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

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Replacing values from column {self._in_col!r} and "
            f"saving output in {self._out_col!r} ..."
        )
        return frame.with_columns(pl.col(self._in_col).replace(**self._kwargs).alias(self._out_col))


class InplaceReplaceTransformer(ReplaceTransformer):
    r"""Replace the values in a column by the values in a mapping.

    Args:
        col: The column name.
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
    >>> from grizz.transformer import InplaceReplace
    >>> transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    >>> transformer
    InplaceReplaceTransformer(col='col', missing_policy='raise', old={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
    >>> frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    >>> frame
    shape: (5, 1)
    ┌─────┐
    │ col │
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
    shape: (5, 1)
    ┌─────┐
    │ col │
    │ --- │
    │ str │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    │ 4   │
    │ 5   │
    └─────┘
    >>> transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3}, default=None)
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 1)
    ┌──────┐
    │ col  │
    │ ---  │
    │ i64  │
    ╞══════╡
    │ 1    │
    │ 2    │
    │ 3    │
    │ null │
    │ null │
    └──────┘

    ```
    """

    def __init__(self, col: str, missing_policy: str = "raise", **kwargs: Any) -> None:
        super().__init__(
            in_col=col,
            out_col=col,
            exist_policy="ignore",
            missing_policy=missing_policy,
            **kwargs,
        )

    def get_args(self) -> dict:
        args = {"col": self._in_col} | super().get_args()
        for key in ["in_col", "out_col", "exist_policy"]:
            args.pop(key)
        return args


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

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Replacing values from column {self._in_col!r} and "
            f"saving output in {self._out_col!r} ..."
        )
        return frame.with_columns(
            pl.col(self._in_col).replace_strict(**self._kwargs).alias(self._out_col)
        )


class InplaceReplaceStrictTransformer(ReplaceStrictTransformer):
    r"""Replace the values in a column by the values in a mapping.

    Args:
        col: The column name.
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
    >>> from grizz.transformer import InplaceReplaceStrict
    >>> transformer = InplaceReplaceStrict(
    ...     col="col", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    ... )
    >>> transformer
    InplaceReplaceStrictTransformer(col='col', missing_policy='raise', old={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
    >>> frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    >>> frame
    shape: (5, 1)
    ┌─────┐
    │ col │
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
    shape: (5, 1)
    ┌─────┐
    │ col │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    │ 4   │
    │ 5   │
    └─────┘
    >>> transformer = InplaceReplaceStrict(
    ...     col="col", old={"a": 1, "b": 2, "c": 3}, default=None
    ... )
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 1)
    ┌──────┐
    │ col  │
    │ ---  │
    │ i64  │
    ╞══════╡
    │ 1    │
    │ 2    │
    │ 3    │
    │ null │
    │ null │
    └──────┘

    ```
    """

    def __init__(self, col: str, missing_policy: str = "raise", **kwargs: Any) -> None:
        super().__init__(
            in_col=col,
            out_col=col,
            exist_policy="ignore",
            missing_policy=missing_policy,
            **kwargs,
        )

    def get_args(self) -> dict:
        args = {"col": self._in_col} | super().get_args()
        for key in ["in_col", "out_col", "exist_policy"]:
            args.pop(key)
        return args
