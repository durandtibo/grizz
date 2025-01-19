# noqa: A005
r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = ["ToDatetimeTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs

from grizz.transformer.columns import BaseInNOutNTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class ToDatetimeTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to convert some columns to a
    ``polars.Datetime`` type.

    Args:
        columns: The columns of type to convert. ``None`` means
            all the columns.
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
        **kwargs: The keyword arguments for ``to_datetime``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ToDatetime
    >>> transformer = ToDatetime(columns=["col1"], prefix="", suffix="_out")
    >>> transformer
    ToDatetimeTransformer(columns=('col1',), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [
    ...             "2020-01-01 01:01:01",
    ...             "2020-01-01 02:02:02",
    ...             "2020-01-01 12:00:01",
    ...             "2020-01-01 18:18:18",
    ...             "2020-01-01 23:59:59",
    ...         ],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": [
    ...             "2020-01-01 11:11:11",
    ...             "2020-02-01 12:12:12",
    ...             "2020-03-01 13:13:13",
    ...             "2020-04-01 08:08:08",
    ...             "2020-05-01 23:59:59",
    ...         ],
    ...     },
    ... )
    >>> frame
    shape: (5, 3)
    ┌─────────────────────┬──────┬─────────────────────┐
    │ col1                ┆ col2 ┆ col3                │
    │ ---                 ┆ ---  ┆ ---                 │
    │ str                 ┆ str  ┆ str                 │
    ╞═════════════════════╪══════╪═════════════════════╡
    │ 2020-01-01 01:01:01 ┆ 1    ┆ 2020-01-01 11:11:11 │
    │ 2020-01-01 02:02:02 ┆ 2    ┆ 2020-02-01 12:12:12 │
    │ 2020-01-01 12:00:01 ┆ 3    ┆ 2020-03-01 13:13:13 │
    │ 2020-01-01 18:18:18 ┆ 4    ┆ 2020-04-01 08:08:08 │
    │ 2020-01-01 23:59:59 ┆ 5    ┆ 2020-05-01 23:59:59 │
    └─────────────────────┴──────┴─────────────────────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌─────────────────────┬──────┬─────────────────────┬─────────────────────┐
    │ col1                ┆ col2 ┆ col3                ┆ col1_out            │
    │ ---                 ┆ ---  ┆ ---                 ┆ ---                 │
    │ str                 ┆ str  ┆ str                 ┆ datetime[μs]        │
    ╞═════════════════════╪══════╪═════════════════════╪═════════════════════╡
    │ 2020-01-01 01:01:01 ┆ 1    ┆ 2020-01-01 11:11:11 ┆ 2020-01-01 01:01:01 │
    │ 2020-01-01 02:02:02 ┆ 2    ┆ 2020-02-01 12:12:12 ┆ 2020-01-01 02:02:02 │
    │ 2020-01-01 12:00:01 ┆ 3    ┆ 2020-03-01 13:13:13 ┆ 2020-01-01 12:00:01 │
    │ 2020-01-01 18:18:18 ┆ 4    ┆ 2020-04-01 08:08:08 ┆ 2020-01-01 18:18:18 │
    │ 2020-01-01 23:59:59 ┆ 5    ┆ 2020-05-01 23:59:59 ┆ 2020-01-01 23:59:59 │
    └─────────────────────┴──────┴─────────────────────┴─────────────────────┘

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
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Converting {len(self.find_columns(frame)):,} columns to time columns...")
        columns = self.find_common_columns(frame)
        frame = frame.select(cs.by_name(columns))
        return frame.select(cs.datetime()).with_columns(
            frame.select((~cs.datetime()).str.to_datetime(**self._kwargs))
        )
