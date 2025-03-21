r"""Contain ``polars.DataFrame`` transformers to scale each column using
statistics that are robust to outliers."""

from __future__ import annotations

__all__ = ["InplaceRobustScalerTransformer", "RobustScalerTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from grizz.transformer.columns import BaseInNOutNTransformer
from grizz.utils.imports import check_sklearn, is_sklearn_available
from grizz.utils.null import propagate_nulls

if is_sklearn_available():  # pragma: no cover
    import sklearn

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class RobustScalerTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to scale each column using statistics
    that are robust to outliers.

    Args:
        columns: The columns to scale. ``None`` means all the
            columns.
        prefix: The column name prefix for the output columns.
        suffix: The column name suffix for the output columns.
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
            ``sklearn.preprocessing.RobustScaler``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import RobustScaler
    >>> transformer = RobustScaler(columns=["col1", "col3"], prefix="", suffix="_out")
    >>> transformer
    RobustScalerTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', propagate_nulls=True)
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
    ┌──────┬──────┬──────┬──────┬──────────┬──────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_out ┆ col3_out │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---      ┆ ---      │
    │ i64  ┆ str  ┆ i64  ┆ str  ┆ f64      ┆ f64      │
    ╞══════╪══════╪══════╪══════╪══════════╪══════════╡
    │ 0    ┆ 0    ┆ 0    ┆ a    ┆ -1.0     ┆ -1.0     │
    │ 1    ┆ 1    ┆ 10   ┆ b    ┆ -0.6     ┆ -0.6     │
    │ 2    ┆ 2    ┆ 20   ┆ c    ┆ -0.2     ┆ -0.2     │
    │ 3    ┆ 3    ┆ 30   ┆ d    ┆ 0.2      ┆ 0.2      │
    │ 4    ┆ 4    ┆ 40   ┆ e    ┆ 0.6      ┆ 0.6      │
    │ 5    ┆ 5    ┆ 50   ┆ f    ┆ 1.0      ┆ 1.0      │
    └──────┴──────┴──────┴──────┴──────────┴──────────┘

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
            prefix=prefix,
            suffix=suffix,
            exclude_columns=exclude_columns,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._propagate_nulls = propagate_nulls

        check_sklearn()
        self._scaler = sklearn.preprocessing.RobustScaler(**kwargs)
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return super().get_args() | {"propagate_nulls": self._propagate_nulls} | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:
        columns = self.find_common_columns(frame)
        logger.info(f"Fitting the robust scaling parameters of {len(columns):,} columns...")
        self._scaler.fit(frame.select(columns).to_numpy())

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        columns = self.find_common_columns(frame)
        logger.info(
            f"Applying the robust scaling transformation on {len(columns):,} "
            f"columns | prefix={self._prefix!r} | suffix={self._suffix!r}"
        )
        data = frame.select(columns)

        x = self._scaler.transform(data.to_numpy())
        out = pl.from_numpy(x, schema=data.columns)
        if self._propagate_nulls:
            out = propagate_nulls(out, data)
        return out


class InplaceRobustScalerTransformer(RobustScalerTransformer):
    r"""Implement a transformer to scale each column using statistics
    that are robust to outliers.

    Args:
        columns: The columns to scale. ``None`` means all the
            columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
        propagate_nulls: If set to ``True``, the ``None`` values are
            propagated after the transformation. If ``False``, the
            ``None`` values are replaced by NaNs.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message appears.
        **kwargs: Additional arguments passed to
            ``sklearn.preprocessing.RobustScaler``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import InplaceRobustScaler
    >>> transformer = InplaceRobustScaler(columns=["col1", "col3"])
    >>> transformer
    InplaceRobustScalerTransformer(columns=('col1', 'col3'), exclude_columns=(), missing_policy='raise', propagate_nulls=True)
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
    shape: (6, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ str  ┆ f64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ -1.0 ┆ 0    ┆ -1.0 ┆ a    │
    │ -0.6 ┆ 1    ┆ -0.6 ┆ b    │
    │ -0.2 ┆ 2    ┆ -0.2 ┆ c    │
    │ 0.2  ┆ 3    ┆ 0.2  ┆ d    │
    │ 0.6  ┆ 4    ┆ 0.6  ┆ e    │
    │ 1.0  ┆ 5    ┆ 1.0  ┆ f    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        exclude_columns: Sequence[str] = (),
        propagate_nulls: bool = True,
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
            propagate_nulls=propagate_nulls,
            **kwargs,
        )

    def get_args(self) -> dict:
        args = super().get_args()
        for key in ["prefix", "suffix", "exist_policy"]:
            args.pop(key)
        return args
