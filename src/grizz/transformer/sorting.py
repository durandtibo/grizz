r"""Contain ``polars.DataFrame`` transformers to sort the DataFrame."""

from __future__ import annotations

__all__ = ["SortColumnsTransformer", "SortTransformer"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_mapping_line

from grizz.transformer.base import BaseTransformer
from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.format import str_kwargs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


class SortTransformer(BaseInNTransformer):
    r"""Implement a transformer to sort the DataFrame by the given
    columns.

    Args:
        columns: The columns to use to sort the rows.
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
        **kwargs: The keyword arguments to pass to ``sort``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import Sort
    >>> transformer = Sort(columns=["col3", "col1"])
    >>> transformer
    SortTransformer(columns=('col3', 'col1'), exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {"col1": [1, 2, None], "col2": [6.0, 5.0, 4.0], "col3": ["a", "c", "b"]}
    ... )
    >>> frame
    shape: (3, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ str  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ 6.0  ┆ a    │
    │ 2    ┆ 5.0  ┆ c    │
    │ null ┆ 4.0  ┆ b    │
    └──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (3, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ str  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ 6.0  ┆ a    │
    │ null ┆ 4.0  ┆ b    │
    │ 2    ┆ 5.0  ┆ c    │
    └──────┴──────┴──────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
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
        cols = self.find_columns(frame)
        logger.info(f"Sorting rows based on {len(cols):,} columns: {cols}")
        # Note: it is not possible to use find_common_columns because find_common_columns
        # may change the order of the columns.
        columns = self._find_existing_columns(frame)
        return frame.sort(columns, **self._kwargs)

    def _find_existing_columns(self, frame: pl.DataFrame) -> list[str]:
        cols = self.find_columns(frame)
        return [col for col in cols if col in frame]


class SortColumnsTransformer(BaseTransformer):
    r"""Implement a transformer to sort the DataFrame columns by name.

    Args:
        reverse: If set to ``False``, then the columns are sorted by
            alphabetical order.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import SortColumns
    >>> transformer = SortColumns()
    >>> transformer
    SortColumnsTransformer(reverse=False)
    >>> frame = pl.DataFrame(
    ...     {"col2": [1, 2, None], "col3": [6.0, 5.0, 4.0], "col1": ["a", "c", "b"]}
    ... )
    >>> frame
    shape: (3, 3)
    ┌──────┬──────┬──────┐
    │ col2 ┆ col3 ┆ col1 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ str  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ 6.0  ┆ a    │
    │ 2    ┆ 5.0  ┆ c    │
    │ null ┆ 4.0  ┆ b    │
    └──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (3, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ str  ┆ i64  ┆ f64  │
    ╞══════╪══════╪══════╡
    │ a    ┆ 1    ┆ 6.0  │
    │ c    ┆ 2    ┆ 5.0  │
    │ b    ┆ null ┆ 4.0  │
    └──────┴──────┴──────┘

    ```
    """

    def __init__(self, reverse: bool = False) -> None:
        self._reverse = reverse

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(reverse={self._reverse})"

    def fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Sorting columns | reverse={self._reverse} ...")
        return frame.select(sorted(frame.columns, reverse=self._reverse))
