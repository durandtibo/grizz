r"""Contain ``polars.DataFrame`` transformers to encode the labels in a
given column."""

from __future__ import annotations

__all__ = ["LabelEncoderTransformer"]

import logging

import polars as pl

from grizz.transformer.columns import BaseIn1Out1Transformer
from grizz.utils.imports import check_sklearn, is_sklearn_available

if is_sklearn_available():  # pragma: no cover
    from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class LabelEncoderTransformer(BaseIn1Out1Transformer):
    r"""Implement a ``polars.DataFrame`` to encode the labels in a given
    column.

    Args:
        in_col: The input column name i.e. the column with the label
            to encode.
        out_col: The output column name i.e. the column with encoded
            labels.
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
    >>> from grizz.transformer import LabelEncoderTransformer
    >>> transformer = LabelEncoderTransformer(in_col="col1", out_col="out")
    >>> transformer
    LabelEncoderTransformer(in_col='col1', out_col='out', exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["a", "b", "c", "d", "e"],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...     }
    ... )
    >>> frame
    shape: (5, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ str  ┆ str  │
    ╞══════╪══════╡
    │ a    ┆ 1    │
    │ b    ┆ 2    │
    │ c    ┆ 3    │
    │ d    ┆ 4    │
    │ e    ┆ 5    │
    └──────┴──────┘
    >>> out = transformer.fit_transform(frame)
    >>> out
    shape: (5, 3)
    ┌──────┬──────┬─────┐
    │ col1 ┆ col2 ┆ out │
    │ ---  ┆ ---  ┆ --- │
    │ str  ┆ str  ┆ i64 │
    ╞══════╪══════╪═════╡
    │ a    ┆ 1    ┆ 0   │
    │ b    ┆ 2    ┆ 1   │
    │ c    ┆ 3    ┆ 2   │
    │ d    ┆ 4    ┆ 3   │
    │ e    ┆ 5    ┆ 4   │
    └──────┴──────┴─────┘

    ```
    """

    def __init__(
        self,
        in_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(
            in_col=in_col,
            out_col=out_col,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )

        check_sklearn()
        self._encoder = LabelEncoder()

    def _fit(self, frame: pl.DataFrame) -> None:
        logger.info(f"Fitting the label encoder to the data in column {self._in_col!r}")
        self._encoder.fit(frame[self._in_col].to_numpy())

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Encoding labels in {self._in_col!r} and saving output in {self._out_col!r} ..."
        )
        y = self._encoder.transform(frame[self._in_col].to_numpy())
        return frame.with_columns(pl.from_numpy(y, schema=[self._out_col]))
