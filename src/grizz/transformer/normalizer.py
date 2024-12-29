r"""Contain ``polars.DataFrame`` transformers to normalize data points
individually to unit norm."""

from __future__ import annotations

__all__ = ["NormalizerTransformer"]

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


class NormalizerTransformer(BaseInNTransformer):
    r"""Implement a transformer to normalize data points individually to
    unit norm.

    Args:
        columns: The columns to scale. ``None`` means all the
            columns.
        prefix: The column name prefix for the copied columns.
        suffix: The column name suffix for the copied columns.
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
        **kwargs: Additional arguments passed to
            ``sklearn.preprocessing.Normalizer``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import Normalizer
    >>> transformer = Normalizer(columns=["col1", "col3"], prefix="", suffix="_norm")
    >>> transformer
    NormalizerTransformer(columns=('col1', 'col3'), prefix='', suffix='_norm', exclude_columns=(), exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 2, 3, 4, 5],
    ...         "col2": ["0", "1", "2", "3", "4", "5"],
    ...         "col3": [5, 4, 3, 2, 1, 0],
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
    │ 0    ┆ 0    ┆ 5    ┆ a    │
    │ 1    ┆ 1    ┆ 4    ┆ b    │
    │ 2    ┆ 2    ┆ 3    ┆ c    │
    │ 3    ┆ 3    ┆ 2    ┆ d    │
    │ 4    ┆ 4    ┆ 1    ┆ e    │
    │ 5    ┆ 5    ┆ 0    ┆ f    │
    └──────┴──────┴──────┴──────┘

    >>> out = transformer.fit_transform(frame)
    >>> out
    shape: (6, 6)
    ┌──────┬──────┬──────┬──────┬───────────┬───────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_norm ┆ col3_norm │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---       ┆ ---       │
    │ i64  ┆ str  ┆ i64  ┆ str  ┆ f64       ┆ f64       │
    ╞══════╪══════╪══════╪══════╪═══════════╪═══════════╡
    │ 0    ┆ 0    ┆ 5    ┆ a    ┆ 0.0       ┆ 1.0       │
    │ 1    ┆ 1    ┆ 4    ┆ b    ┆ 0.242536  ┆ 0.970143  │
    │ 2    ┆ 2    ┆ 3    ┆ c    ┆ 0.5547    ┆ 0.83205   │
    │ 3    ┆ 3    ┆ 2    ┆ d    ┆ 0.83205   ┆ 0.5547    │
    │ 4    ┆ 4    ┆ 1    ┆ e    ┆ 0.970143  ┆ 0.242536  │
    │ 5    ┆ 5    ┆ 0    ┆ f    ┆ 1.0       ┆ 0.0       │
    └──────┴──────┴──────┴──────┴───────────┴───────────┘

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
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._prefix = prefix
        self._suffix = suffix

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy

        check_sklearn()
        self._scaler = sklearn.preprocessing.Normalizer(**kwargs)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "prefix": self._prefix,
                "suffix": self._suffix,
                "exclude_columns": self._exclude_columns,
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
        self._check_output_columns(frame)
        logger.info(
            f"Binarize the data of {len(self.find_columns(frame)):,} columns | "
            f"prefix={self._prefix!r} | suffix={self._suffix!r}"
        )
        columns = self.find_common_columns(frame)
        data = frame.select(columns).fill_nan(None)

        x = self._scaler.transform(data.fill_null(0).to_numpy())
        data_norm = pl.from_numpy(x, schema=data.columns)
        data_norm = propagate_nulls(data_norm, data)
        return frame.with_columns(
            data_norm.rename(lambda col: f"{self._prefix}{col}{self._suffix}")
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
