r"""Contain ``polars.DataFrame`` transformers to apply a power transform
feature-wise to make data more Gaussian-like."""

from __future__ import annotations

__all__ = ["PowerTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from grizz.transformer.columns import BaseInNTransformer
from grizz.utils.column import check_column_exist_policy, check_existing_columns
from grizz.utils.imports import check_sklearn, is_sklearn_available
from grizz.utils.null import propagate_nulls

if is_sklearn_available():  # pragma: no cover
    import sklearn

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class PowerTransformer(BaseInNTransformer):
    r"""Implement a transformer to apply a power transform featurewise to
    make data more Gaussian-like.

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
            ``sklearn.preprocessing.PowerTransformer``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import PowerTransformer
    >>> transformer = PowerTransformer(columns=["col1", "col3"], prefix="", suffix="_out")
    >>> transformer
    PowerTransformer(columns=('col1', 'col3'), exclude_columns=(), missing_policy='raise', exist_policy='raise', propagate_nulls=True, prefix='', suffix='_out')
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
    ┌──────┬──────┬──────┬──────┬───────────┬───────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_out  ┆ col3_out  │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---       ┆ ---       │
    │ i64  ┆ str  ┆ i64  ┆ str  ┆ f64       ┆ f64       │
    ╞══════╪══════╪══════╪══════╪═══════════╪═══════════╡
    │ 0    ┆ 0    ┆ 0    ┆ a    ┆ -1.567837 ┆ -1.695398 │
    │ 1    ┆ 1    ┆ 10   ┆ b    ┆ -0.836194 ┆ -0.740367 │
    │ 2    ┆ 2    ┆ 20   ┆ c    ┆ -0.210053 ┆ -0.117399 │
    │ 3    ┆ 3    ┆ 30   ┆ d    ┆ 0.356111  ┆ 0.402585  │
    │ 4    ┆ 4    ┆ 40   ┆ e    ┆ 0.881486  ┆ 0.864187  │
    │ 5    ┆ 5    ┆ 50   ┆ f    ┆ 1.376486  ┆ 1.286392  │
    └──────┴──────┴──────┴──────┴───────────┴───────────┘

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
        self._transformer = sklearn.preprocessing.PowerTransformer(**kwargs)
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return (
            super().get_args()
            | {
                "exist_policy": self._exist_policy,
                "propagate_nulls": self._propagate_nulls,
                "prefix": self._prefix,
                "suffix": self._suffix,
            }
            | self._kwargs
        )

    def _fit(self, frame: pl.DataFrame) -> None:
        logger.info(
            f"Fitting the power transformation on {len(self.find_columns(frame)):,} columns..."
        )
        columns = self.find_common_columns(frame)
        self._transformer.fit(frame.select(columns).to_numpy())

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_output_columns(frame)
        logger.info(
            f"Applying the power transformation on {len(self.find_columns(frame)):,} columns | "
            f"prefix={self._prefix!r} | suffix={self._suffix!r}"
        )
        columns = self.find_common_columns(frame)
        data = frame.select(columns)

        x = self._transformer.transform(data.to_numpy())
        data_out = pl.from_numpy(x, schema=data.columns)
        if self._propagate_nulls:
            data_out = propagate_nulls(data_out, data)
        return frame.with_columns(data_out.rename(lambda col: f"{self._prefix}{col}{self._suffix}"))

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
