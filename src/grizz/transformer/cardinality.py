r"""Contain transformers to filter based on the cardinality (i.e. number
of unique values) in each column."""

from __future__ import annotations

__all__ = ["FilterCardinalityTransformer"]

import logging
from typing import TYPE_CHECKING

import polars as pl
from coola.utils.format import repr_mapping_line

from grizz.transformer.columns import BaseColumnsTransformer
from grizz.utils.format import str_col_diff

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class FilterCardinalityTransformer(BaseColumnsTransformer):
    r"""Implement a transformer to filter based on the cardinality (i.e.
    number of unique values) in each column.

    Args:
        columns: The columns to use to filter based on the number of
            unique values. If ``None``, it processes all the columns
            of type string.
        n_min: The minimal cardinality (included).
        n_max: The maximal cardinality (excluded).
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
    >>> from grizz.transformer import FilterCardinality
    >>> transformer = FilterCardinality(columns=["col1", "col2", "col3"], n_min=2, n_max=5)
    >>> transformer
    FilterCardinalityTransformer(columns=('col1', 'col2', 'col3'), n_min=2, n_max=5, missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": [1, 1, 1, 1, 1],
    ...         "col3": ["a", "b", "c", "a", "b"],
    ...         "col4": [1.2, float("nan"), 3.2, None, 5.2],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ str  ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ a    ┆ 1.2  │
    │ 2    ┆ 1    ┆ b    ┆ NaN  │
    │ 3    ┆ 1    ┆ c    ┆ 3.2  │
    │ 4    ┆ 1    ┆ a    ┆ null │
    │ 5    ┆ 1    ┆ b    ┆ 5.2  │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 2)
    ┌──────┬──────┐
    │ col3 ┆ col4 │
    │ ---  ┆ ---  │
    │ str  ┆ f64  │
    ╞══════╪══════╡
    │ a    ┆ 1.2  │
    │ b    ┆ NaN  │
    │ c    ┆ 3.2  │
    │ a    ┆ null │
    │ b    ┆ 5.2  │
    └──────┴──────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        n_min: int = 0,
        n_max: int = float("inf"),
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(columns=columns, missing_policy=missing_policy)
        self._n_min = n_min
        self._n_max = n_max

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "n_min": self._n_min,
                "n_max": self._n_max,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Filtering {len(self.find_columns(frame)):,} columns based on their "
            f"cardinality [{self._n_min}, {self._n_max})..."
        )
        self._check_input_columns(frame)
        initial_shape = frame.shape
        columns = self.find_common_columns(frame)
        valid = frame.select(
            pl.n_unique(*columns).is_between(self._n_min, self._n_max, closed="left")
        )
        cols_to_drop = [col.name for col in valid.iter_columns() if not col.first()]
        logger.info(f"Dropping {len(cols_to_drop):,} columns: {cols_to_drop}")
        out = frame.drop(cols_to_drop)
        logger.info(
            f"DataFrame shape: {initial_shape} -> {out.shape} | "
            f"{str_col_diff(orig=initial_shape[1], final=out.shape[1])}"
        )
        return out
