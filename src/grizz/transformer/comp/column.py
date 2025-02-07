r"""Contain ``polars.DataFrame`` transformers to compare element-wise
two columns of a DataFrame."""

from __future__ import annotations

__all__ = [
    "BaseColumnComparatorTransformer",
    "ColumnEqualMissingTransformer",
    "ColumnEqualTransformer",
    "ColumnGreaterEqualTransformer",
    "ColumnGreaterTransformer",
    "ColumnLowerEqualTransformer",
    "ColumnLowerTransformer",
    "ColumnNotEqualMissingTransformer",
    "ColumnNotEqualTransformer",
]

import logging

import polars as pl

from grizz.transformer.columns import BaseIn2Out1Transformer

logger = logging.getLogger(__name__)


class BaseColumnComparatorTransformer(BaseIn2Out1Transformer):
    r"""Define a base class to compare element-wise two columns of a
    DataFrame."""

    def _fit(self, frame: pl.DataFrame) -> None:  # noqa: ARG002
        logger.info(
            f"Skipping '{self.__class__.__qualname__}.fit' as there are no parameters "
            f"available to fit"
        )


class ColumnEqualTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the equal operation between
    two columns (``in1 == in2``).

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnEqual
    >>> transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the equal operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).eq(pl.col(self._in2_col)).alias(self._out_col)
        )


class ColumnEqualMissingTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the equal operation between
    two columns (``in1 == in2``), where null values are not propagated.

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnEqualMissing
    >>> transformer = ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnEqualMissingTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the equal missing operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).eq_missing(pl.col(self._in2_col)).alias(self._out_col)
        )


class ColumnGreaterEqualTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the greater than or equal
    operation between two columns (``in1 >= in2``).

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnGreaterEqual
    >>> transformer = ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnGreaterEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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
    │ 4    ┆ 2    ┆ d    ┆ true  │
    │ 5    ┆ 1    ┆ e    ┆ true  │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the greater than or equal operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).ge(pl.col(self._in2_col)).alias(self._out_col)
        )


class ColumnGreaterTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the greater than operation
    between two columns (``in1 > in2``).

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnGreater
    >>> transformer = ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnGreaterTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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
    │ 3    ┆ 3    ┆ c    ┆ false │
    │ 4    ┆ 2    ┆ d    ┆ true  │
    │ 5    ┆ 1    ┆ e    ┆ true  │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the greater than operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).gt(pl.col(self._in2_col)).alias(self._out_col)
        )


class ColumnLowerEqualTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the lower than or equal
    operation between two columns (``in1 <= in2``).

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnLowerEqual
    >>> transformer = ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnLowerEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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
    │ 1    ┆ 5    ┆ a    ┆ true  │
    │ 2    ┆ 4    ┆ b    ┆ true  │
    │ 3    ┆ 3    ┆ c    ┆ true  │
    │ 4    ┆ 2    ┆ d    ┆ false │
    │ 5    ┆ 1    ┆ e    ┆ false │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the lower than or equal operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).le(pl.col(self._in2_col)).alias(self._out_col)
        )


class ColumnLowerTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the lower than operation
    between two columns (``in1 < in2``).

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnLower
    >>> transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnLowerTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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
    │ 1    ┆ 5    ┆ a    ┆ true  │
    │ 2    ┆ 4    ┆ b    ┆ true  │
    │ 3    ┆ 3    ┆ c    ┆ false │
    │ 4    ┆ 2    ┆ d    ┆ false │
    │ 5    ┆ 1    ┆ e    ┆ false │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the lower than operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).lt(pl.col(self._in2_col)).alias(self._out_col)
        )


class ColumnNotEqualTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the not equal operation
    between two columns (``in1 != in2``).

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnNotEqual
    >>> transformer = ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnNotEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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
    │ 1    ┆ 5    ┆ a    ┆ true  │
    │ 2    ┆ 4    ┆ b    ┆ true  │
    │ 3    ┆ 3    ┆ c    ┆ false │
    │ 4    ┆ 2    ┆ d    ┆ true  │
    │ 5    ┆ 1    ┆ e    ┆ true  │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the not equal operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).ne(pl.col(self._in2_col)).alias(self._out_col)
        )


class ColumnNotEqualMissingTransformer(BaseColumnComparatorTransformer):
    r"""Implement a transformer that computes the not equal operation
    between two columns (``in1 != in2``), where null values are not
    propagated.

    Args:
        in1_col: The first input column name.
        in2_col: The second input column name.
        out_col: The output column name.
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
    >>> from grizz.transformer import ColumnNotEqualMissing
    >>> transformer = ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    >>> transformer
    ColumnNotEqualMissingTransformer(in1_col='col1', in2_col='col2', out_col='out', exist_policy='raise', missing_policy='raise')
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
    │ 1    ┆ 5    ┆ a    ┆ true  │
    │ 2    ┆ 4    ┆ b    ┆ true  │
    │ 3    ┆ 3    ┆ c    ┆ false │
    │ 4    ┆ 2    ┆ d    ┆ true  │
    │ 5    ┆ 1    ┆ e    ┆ true  │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            f"Applying the not equal missing operation between {self._in1_col!r} "
            f"and {self._in2_col!r} | out_col={self._out_col!r}"
        )
        return frame.with_columns(
            pl.col(self._in1_col).ne_missing(pl.col(self._in2_col)).alias(self._out_col)
        )
