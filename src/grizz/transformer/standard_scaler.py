r"""Contain ``polars.DataFrame`` transformers to standardize each column
by removing the mean and scaling to unit variance."""

from __future__ import annotations

__all__ = ["StandardScalerTransformer"]

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


class StandardScalerTransformer(BaseInNTransformer):
    r"""Implement a transformer to standardize each column by removing
    the mean and scaling to unit variance.

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
            ``sklearn.preprocessing.StandardScaler``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import StandardScaler
    >>> transformer = StandardScaler(columns=["col1", "col3"], prefix="", suffix="_scaled")
    >>> transformer
    StandardScalerTransformer(columns=('col1', 'col3'), prefix='', suffix='_scaled', exclude_columns=(), propagate_nulls=True, exist_policy='raise', missing_policy='raise')
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
    ┌──────┬──────┬──────┬──────┬─────────────┬─────────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_scaled ┆ col3_scaled │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---         ┆ ---         │
    │ i64  ┆ str  ┆ i64  ┆ str  ┆ f64         ┆ f64         │
    ╞══════╪══════╪══════╪══════╪═════════════╪═════════════╡
    │ 1    ┆ 1    ┆ 10   ┆ a    ┆ -1.414214   ┆ -1.414214   │
    │ 2    ┆ 2    ┆ 20   ┆ b    ┆ -0.707107   ┆ -0.707107   │
    │ 3    ┆ 3    ┆ 30   ┆ c    ┆ 0.0         ┆ 0.0         │
    │ 4    ┆ 4    ┆ 40   ┆ d    ┆ 0.707107    ┆ 0.707107    │
    │ 5    ┆ 5    ┆ 50   ┆ e    ┆ 1.414214    ┆ 1.414214    │
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
        self._scaler = sklearn.preprocessing.StandardScaler(**kwargs)
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
            f"Fitting the robust scaling parameters of {len(self.find_columns(frame)):,} "
            "columns..."
        )
        columns = self.find_common_columns(frame)
        self._scaler.fit(frame.select(columns).to_numpy())

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_output_columns(frame)
        logger.info(
            f"Applying the robust scaling transformation on {len(self.find_columns(frame)):,} "
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
