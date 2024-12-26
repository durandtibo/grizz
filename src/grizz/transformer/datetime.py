r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = ["ToDatetimeTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_kwargs

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class ToDatetimeTransformer(BaseInNTransformer):
    r"""Implement a transformer to convert some columns to a
    ``polars.Datetime`` type.

    Args:
        columns: The columns to convert. ``None`` means all the
            columns.
        format: Format to use for conversion. Refer to the
            [chrono crate documentation](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)
            for the full specification.
            Example: ``"%Y-%m-%d %H:%M:%S"``.
            If set to ``None`` (default), the format is inferred from
            the data.
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
        **kwargs: The keyword arguments for ``to_datetime``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ToDatetime
    >>> transformer = ToDatetime(columns=["col1"])
    >>> transformer
    ToDatetimeTransformer(columns=('col1',), format=None, exclude_columns=(), missing_policy='raise')
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
    shape: (5, 3)
    ┌─────────────────────┬──────┬─────────────────────┐
    │ col1                ┆ col2 ┆ col3                │
    │ ---                 ┆ ---  ┆ ---                 │
    │ datetime[μs]        ┆ str  ┆ str                 │
    ╞═════════════════════╪══════╪═════════════════════╡
    │ 2020-01-01 01:01:01 ┆ 1    ┆ 2020-01-01 11:11:11 │
    │ 2020-01-01 02:02:02 ┆ 2    ┆ 2020-02-01 12:12:12 │
    │ 2020-01-01 12:00:01 ┆ 3    ┆ 2020-03-01 13:13:13 │
    │ 2020-01-01 18:18:18 ┆ 4    ┆ 2020-04-01 08:08:08 │
    │ 2020-01-01 23:59:59 ┆ 5    ┆ 2020-05-01 23:59:59 │
    └─────────────────────┴──────┴─────────────────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        format: str | None = None,  # noqa: A002
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._format = format
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "format": self._format,
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
        logger.info(
            f"Converting {len(self.find_columns(frame)):,} columns to datetime "
            f"| {self._format!r} ..."
        )
        columns = self.find_common_columns(frame)
        return frame.with_columns(
            frame.select(
                (cs.by_name(columns) & cs.string()).str.to_datetime(self._format, **self._kwargs)
            )
        )
