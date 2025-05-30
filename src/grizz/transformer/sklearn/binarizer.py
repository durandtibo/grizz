r"""Contain ``polars.DataFrame`` transformers to binarize data according
to a threshold."""

from __future__ import annotations

__all__ = ["BinarizerTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from grizz.transformer.columns import BaseInNOutNTransformer
from grizz.transformer.utils import get_classname, message_skip_fit
from grizz.utils.imports import check_sklearn, is_sklearn_available
from grizz.utils.null import propagate_nulls

if is_sklearn_available():  # pragma: no cover
    import sklearn

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class BinarizerTransformer(BaseInNOutNTransformer):
    r"""Implement a transformer to binarize data according to a
    threshold.

    Args:
        columns: The columns to scale. ``None`` means all the
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
        **kwargs: Additional arguments passed to
            ``sklearn.preprocessing.Binarizer``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import Binarizer
    >>> transformer = Binarizer(
    ...     columns=["col1", "col3"], prefix="", suffix="_out", threshold=1.5
    ... )
    >>> transformer
    BinarizerTransformer(columns=('col1', 'col3'), exclude_columns=(), exist_policy='raise', missing_policy='raise', prefix='', suffix='_out', threshold=1.5)
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
    ┌──────┬──────┬──────┬──────┬──────────┬──────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col1_out ┆ col3_out │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---      ┆ ---      │
    │ i64  ┆ str  ┆ i64  ┆ str  ┆ i64      ┆ i64      │
    ╞══════╪══════╪══════╪══════╪══════════╪══════════╡
    │ 0    ┆ 0    ┆ 5    ┆ a    ┆ 0        ┆ 1        │
    │ 1    ┆ 1    ┆ 4    ┆ b    ┆ 0        ┆ 1        │
    │ 2    ┆ 2    ┆ 3    ┆ c    ┆ 1        ┆ 1        │
    │ 3    ┆ 3    ┆ 2    ┆ d    ┆ 1        ┆ 1        │
    │ 4    ┆ 4    ┆ 1    ┆ e    ┆ 1        ┆ 0        │
    │ 5    ┆ 5    ┆ 0    ┆ f    ┆ 1        ┆ 0        │
    └──────┴──────┴──────┴──────┴──────────┴──────────┘

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

        check_sklearn()
        self._scaler = sklearn.preprocessing.Binarizer(**kwargs)
        self._kwargs = kwargs

    def get_args(self) -> dict:
        return super().get_args() | self._kwargs

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(message_skip_fit(get_classname(self)))

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        columns = self.find_common_columns(frame)
        logger.info(
            f"Binarize the data of {len(columns):,} columns | "
            f"prefix={self._prefix!r} | suffix={self._suffix!r}"
        )
        data = frame.select(columns).fill_nan(None)

        x = self._scaler.transform(data.fill_null(0).to_numpy())
        out = pl.from_numpy(x, schema=data.columns)
        return propagate_nulls(out, data)
