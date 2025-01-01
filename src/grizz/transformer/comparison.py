r"""Contain ``polars.DataFrame`` transformers to compare element-wise a
DataFrame."""

from __future__ import annotations

__all__ = ["BaseComparatorTransformer", "GreaterEqualTransformer"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.column import check_column_exist_policy, check_existing_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

logger = logging.getLogger(__name__)


class BaseComparatorTransformer(BaseInNTransformer):
    r"""Define a base class to compare element-wise a DataFrame.

    Args:
        columns: The columns to compare. ``None`` means all the
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
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        target: Any,
        prefix: str,
        suffix: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._target = target
        self._prefix = prefix
        self._suffix = suffix

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "target": self._target,
                "prefix": self._prefix,
                "suffix": self._suffix,
                "exclude_columns": self._exclude_columns,
                "exist_policy": self._exist_policy,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_output_columns(frame)
        logger.info(
            f"Applying the robust scaling transformation on {len(self.find_columns(frame)):,} "
            f"columns | prefix={self._prefix!r} | suffix={self._suffix!r}"
        )
        columns = self.find_common_columns(frame)
        data = self._compare(frame.select(columns))
        return frame.with_columns(data.rename(lambda col: f"{self._prefix}{col}{self._suffix}"))

    def _check_output_columns(self, frame: pl.DataFrame) -> None:
        r"""Check if the output columns already exist.

        Args:
            frame: The input DataFrame to check.
        """
        check_existing_columns(
            frame,
            columns=[f"{self._prefix}{col}{self._suffix}" for col in self.find_columns(frame)],
            exist_policy=self._exist_policy,
        )

    @abstractmethod
    def _compare(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Generate the comparison results.

        Args:
            frame: The DataFrame to compare.

        Returns:
            A DataFrame with the same shape and columns as the input,
                but that contains the result of the comparison.
        """


class GreaterEqualTransformer(BaseComparatorTransformer):
    r"""Implements a transformer that computes the greater than or equal
    operation.

    Args:
        columns: The columns to compare. ``None`` means all the
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

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import GreaterEqual
    >>> transformer = GreaterEqual(
    ...     columns=["col1", "col3"], target=4.2, prefix="", suffix="_ind"
    ... )
    >>> transformer
    GreaterEqualTransformer(columns=('col1', 'col3'), target=4.2, prefix='', suffix='_ind', exclude_columns=(), exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": [10, 20, 30, 40, 50],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 10   ┆ a    │
    │ 2    ┆ 2    ┆ 20   ┆ b    │
    │ 3    ┆ 3    ┆ 30   ┆ c    │
    │ 4    ┆ 4    ┆ 40   ┆ d    │
    │ 5    ┆ 5    ┆ 50   ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.fit_transform(frame)
    >>> out
    shape: (5, 6)
    ┌──────┬──────┬──────┬──────┬──────────┬──────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_ind ┆ col3_ind │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---      ┆ ---      │
    │ i64  ┆ str  ┆ i64  ┆ str  ┆ bool     ┆ bool     │
    ╞══════╪══════╪══════╪══════╪══════════╪══════════╡
    │ 1    ┆ 1    ┆ 10   ┆ a    ┆ false    ┆ true     │
    │ 2    ┆ 2    ┆ 20   ┆ b    ┆ false    ┆ true     │
    │ 3    ┆ 3    ┆ 30   ┆ c    ┆ false    ┆ true     │
    │ 4    ┆ 4    ┆ 40   ┆ d    ┆ false    ┆ true     │
    │ 5    ┆ 5    ┆ 50   ┆ e    ┆ true     ┆ true     │
    └──────┴──────┴──────┴──────┴──────────┴──────────┘

    ```
    """

    def _compare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame >= self._target
