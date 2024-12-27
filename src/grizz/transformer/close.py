r"""Contain a ``polars.DataFrame`` transformer to indicate if the values
of two columns are element-wise equal within a tolerance."""

from __future__ import annotations

__all__ = ["CloseColumnsTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseIn2Out1Transformer
from grizz.utils.format import str_boolean_series_stats

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class CloseColumnsTransformer(BaseIn2Out1Transformer):
    r"""Implement a transformer to compute a column that indicates if the
    values of two columns are element-wise equal within a tolerance.

    The output column contains ``True`` if two columns are element-wise
    equal within a tolerance. Internally, this tranformer computes:
    ``out = (|actual - expected| <= atol + rtol * |expected|)``

    Args:
        actual: The actual input column name. This column must be a
            numeric column.
        expected: The expected input column name. This column must be
            a numeric column.
        out_col: The output column name.
        atol: The absolute tolerance parameter.
        rtol: The relative tolerance parameter.
        equal_nan: Whether to compare NaN's as equal. If ``True``,
            NaN's in ``actual`` will be considered equal to NaN's in
            ``expected`` in the output column.
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
    >>> from grizz.transformer import CloseColumns
    >>> transformer = CloseColumns(actual="col1", expected="col2", out_col="out")
    >>> transformer
    CloseColumnsTransformer(actual='col1', expected='col2', out_col='out', atol=1e-08, rtol=1e-05, equal_nan=False, exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": [5, 4, 3, 2, 1],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ 5    ┆ a    │
    │ 2    ┆ 4    ┆ b    │
    │ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 2    ┆ d    │
    │ 5    ┆ 1    ┆ e    │
    └──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ col3 ┆ out   │
    │ ---  ┆ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ i64  ┆ str  ┆ bool  │
    ╞══════╪══════╪══════╪═══════╡
    │ 1    ┆ 5    ┆ a    ┆ false │
    │ 2    ┆ 4    ┆ b    ┆ false │
    │ 3    ┆ 3    ┆ c    ┆ true  │
    │ 4    ┆ 2    ┆ d    ┆ false │
    │ 5    ┆ 1    ┆ e    ┆ false │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def __init__(
        self,
        actual: str,
        expected: str,
        out_col: str,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        equal_nan: bool = False,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(
            in1_col=actual,
            in2_col=expected,
            out_col=out_col,
            exist_policy=exist_policy,
            missing_policy=missing_policy,
        )
        self._atol = float(atol)
        self._rtol = float(rtol)
        self._equal_nan = equal_nan

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "actual": self._in1_col,
                "expected": self._in2_col,
                "out_col": self._out_col,
                "atol": self._atol,
                "rtol": self._rtol,
                "equal_nan": self._equal_nan,
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
        logger.info(
            f"Computing the equality within tolerance between actual column {self._in1_col!r} "
            f"and expected column {self._in2_col!r} | out_col={self._out_col!r} | "
            f"atol={self._atol}  rtol={self._rtol}  equal_nan={self._equal_nan}"
        )
        diff = (frame[self._in1_col] - frame[self._in2_col]).abs()
        tol = frame[self._in2_col].abs() * self._rtol + self._atol
        tol_check = (diff <= tol) & ~frame[self._in2_col].is_nan()

        if self._equal_nan:
            nan_check = frame[self._in1_col].is_nan() & frame[self._in2_col].is_nan()
            tol_check = tol_check | nan_check
        logger.info(f"column: {self._out_col!r} | {str_boolean_series_stats(tol_check)}")
        return frame.with_columns(tol_check.alias(self._out_col))
