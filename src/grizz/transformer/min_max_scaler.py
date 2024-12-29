r"""Contain ``polars.DataFrame`` transformers to scale each column to a
given range."""

from __future__ import annotations

__all__ = ["MinMaxScalerTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.column import check_column_exist_policy, check_existing_columns
from grizz.utils.format import str_kwargs
from grizz.utils.imports import check_sklearn, is_sklearn_available
from grizz.utils.null import propagate_nulls

if is_sklearn_available():  # pragma: no cover
    import sklearn

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class MinMaxScalerTransformer(BaseInNTransformer):
    r"""Implement a transformer to scale each column to a given range.

    Args:
        columns: The columns to scale. ``None`` means all the
            columns.
        prefix: The column name prefix for the copied columns.
        suffix: The column name suffix for the copied columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
        propagate_nulls: If set to ``True``, the ``None`` values are
            propagated after the transformation. If ``False``, the
            ``None`` values are replaced by NaNs.
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
            ``sklearn.preprocessing.MinMaxScaler``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import MinMaxScaler
    >>> transformer = MinMaxScaler(columns=["col1", "col3"], prefix="", suffix="_scaled")
    >>> transformer
    MinMaxScalerTransformer(columns=('col1', 'col3'), prefix='', suffix='_scaled', exclude_columns=(), propagate_nulls=True, exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 2, 3, 4, 5],
    ...         "col2": ["0", "1", "2", "3", "4", "5"],
    ...         "col3": [0, 10, 20, 30, 40, 50],
    ...         "col4": ["a", "b", "c", "d", "e", "f"],
    ...     }
    ... )
    >>> frame
    shape: (6, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 0    ┆ 0    ┆ 0    ┆ a    │
    │ 1    ┆ 1    ┆ 10   ┆ b    │
    │ 2    ┆ 2    ┆ 20   ┆ c    │
    │ 3    ┆ 3    ┆ 30   ┆ d    │
    │ 4    ┆ 4    ┆ 40   ┆ e    │
    │ 5    ┆ 5    ┆ 50   ┆ f    │
    └──────┴──────┴──────┴──────┘

    >>> out = transformer.fit_transform(frame)
    >>> out
    shape: (6, 6)
    ┌──────┬──────┬──────┬──────┬─────────────┬─────────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_scaled ┆ col3_scaled │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---         ┆ ---         │
    │ i64  ┆ str  ┆ i64  ┆ str  ┆ f64         ┆ f64         │
    ╞══════╪══════╪══════╪══════╪═════════════╪═════════════╡
    │ 0    ┆ 0    ┆ 0    ┆ a    ┆ 0.0         ┆ 0.0         │
    │ 1    ┆ 1    ┆ 10   ┆ b    ┆ 0.2         ┆ 0.2         │
    │ 2    ┆ 2    ┆ 20   ┆ c    ┆ 0.4         ┆ 0.4         │
    │ 3    ┆ 3    ┆ 30   ┆ d    ┆ 0.6         ┆ 0.6         │
    │ 4    ┆ 4    ┆ 40   ┆ e    ┆ 0.8         ┆ 0.8         │
    │ 5    ┆ 5    ┆ 50   ┆ f    ┆ 1.0         ┆ 1.0         │
    └──────┴──────┴──────┴──────┴─────────────┴─────────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        exclude_columns: Sequence[str] = (),
        propagate_nulls: bool = True,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._prefix = prefix
        self._suffix = suffix
        self._propagate_nulls = propagate_nulls

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy

        check_sklearn()
        self._scaler = sklearn.preprocessing.MinMaxScaler(**kwargs)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "prefix": self._prefix,
                "suffix": self._suffix,
                "exclude_columns": self._exclude_columns,
                "propagate_nulls": self._propagate_nulls,
                "exist_policy": self._exist_policy,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args}{str_kwargs(self._kwargs)})"

    def _fit(self, frame: pl.DataFrame) -> None:
        logger.info(
            f"Fitting the min/max scaling parameters of {len(self.find_columns(frame)):,} "
            "columns..."
        )
        columns = self.find_common_columns(frame)
        self._scaler.fit(frame.select(columns).to_numpy())

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_output_columns(frame)
        logger.info(
            f"Applying the min/max scaling transformation on {len(self.find_columns(frame)):,} "
            f"columns | prefix={self._prefix!r} | suffix={self._suffix!r}"
        )
        columns = self.find_common_columns(frame)
        data = frame.select(columns)

        x = self._scaler.transform(data.to_numpy())
        data_scaled = pl.from_numpy(x, schema=data.columns)
        if self._propagate_nulls:
            data_scaled = propagate_nulls(data_scaled, data)
        return frame.with_columns(
            data_scaled.rename(lambda col: f"{self._prefix}{col}{self._suffix}")
        )

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
