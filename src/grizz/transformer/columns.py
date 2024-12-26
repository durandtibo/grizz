r"""Contain a base class to implement ``polars.DataFrame`` transformers
that transform DataFrames by using multiple columns."""

from __future__ import annotations

__all__ = [
    "BaseIn1Out1Transformer",
    "BaseIn2Out1Transformer",
    "BaseInNOut1Transformer",
    "BaseInNTransformer",
]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from coola.utils.format import repr_mapping_line

from grizz.transformer.base import BaseTransformer
from grizz.utils.column import (
    check_column_exist_policy,
    check_column_missing_policy,
    check_existing_column,
    check_missing_column,
    check_missing_columns,
    find_common_columns,
    find_missing_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

logger = logging.getLogger(__name__)


class BaseIn1Out1Transformer(BaseTransformer):
    r"""Define a base class to implement ``polars.DataFrame``
    transformers that takes one input column and generate one output
    column.

    Args:
        in_col: The input column name.
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
    """

    def __init__(
        self,
        in_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        self._in_col = in_col
        self._out_col = out_col

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy
        check_column_missing_policy(missing_policy)
        self._missing_policy = missing_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "in_col": self._in_col,
                "out_col": self._out_col,
                "exist_policy": self._exist_policy,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit(self, frame: pl.DataFrame) -> None:
        self._check_input_column(frame)
        if self._in_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.fit' "
                f"because the input column {self._in_col!r} is missing"
            )
            return
        self._fit(frame)

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_input_column(frame)
        if self._in_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.transform' "
                f"because the input column {self._in_col!r} is missing"
            )
            return frame
        self._check_output_column(frame)
        return self._transform(frame)

    def _check_input_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the input column is missing.

        Args:
            frame: The input DataFrame to check.
        """
        check_missing_column(frame, column=self._in_col, missing_policy=self._missing_policy)

    def _check_output_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input DataFrame to check.
        """
        check_existing_column(frame, column=self._out_col, exist_policy=self._exist_policy)

    @abstractmethod
    def _fit(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Fit to the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.
        """


class BaseIn2Out1Transformer(BaseTransformer):
    r"""Define a base class to implement ``polars.DataFrame``
    transformers that takes two input columns and generate one output
    column.

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
    >>> from grizz.transformer import AbsDiffHorizontal
    >>> transformer = AbsDiffHorizontal(in1_col="col1", in2_col="col2", out_col="diff")
    >>> transformer
    AbsDiffHorizontalTransformer(in1_col='col1', in2_col='col2', out_col='diff', exist_policy='raise', missing_policy='raise')
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
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ diff │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ str  ┆ i64  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 5    ┆ a    ┆ 4    │
    │ 2    ┆ 4    ┆ b    ┆ 2    │
    │ 3    ┆ 3    ┆ c    ┆ 0    │
    │ 4    ┆ 2    ┆ d    ┆ 2    │
    │ 5    ┆ 1    ┆ e    ┆ 4    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def __init__(
        self,
        in1_col: str,
        in2_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        self._in1_col = in1_col
        self._in2_col = in2_col
        self._out_col = out_col

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy
        check_column_missing_policy(missing_policy)
        self._missing_policy = missing_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "in1_col": self._in1_col,
                "in2_col": self._in2_col,
                "out_col": self._out_col,
                "exist_policy": self._exist_policy,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit(self, frame: pl.DataFrame) -> None:
        self._check_input_columns(frame)
        if self._in1_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.fit' "
                f"because the input column {self._in1_col!r} is missing"
            )
            return
        if self._in2_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.fit' "
                f"because the input column {self._in2_col!r} is missing"
            )
            return
        self._fit(frame)

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_input_columns(frame)
        if self._in1_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.transform' "
                f"because the input column {self._in1_col!r} is missing"
            )
            return frame
        if self._in2_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.transform' "
                f"because the input column {self._in2_col!r} is missing"
            )
            return frame
        self._check_output_column(frame)
        return self._transform(frame)

    def _check_input_columns(self, frame: pl.DataFrame) -> None:
        r"""Check if any of the input columns is missing.

        Args:
            frame: The input DataFrame to check.
        """
        check_missing_column(frame, column=self._in1_col, missing_policy=self._missing_policy)
        check_missing_column(frame, column=self._in2_col, missing_policy=self._missing_policy)

    def _check_output_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input DataFrame to check.
        """
        check_existing_column(frame, column=self._out_col, exist_policy=self._exist_policy)

    @abstractmethod
    def _fit(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Fit to the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.
        """


class BaseInNTransformer(BaseTransformer):
    r"""Define a base class to implement ``polars.DataFrame``
    transformers that transform DataFrames by using multiple input
    columns.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
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
    >>> from grizz.transformer import StripChars
    >>> transformer = StripChars(columns=["col2", "col3"])
    >>> transformer
    StripCharsTransformer(columns=('col2', 'col3'), exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬───────┬───────┐
    │ col1 ┆ col2 ┆ col3  ┆ col4  │
    │ ---  ┆ ---  ┆ ---   ┆ ---   │
    │ i64  ┆ str  ┆ str   ┆ str   │
    ╞══════╪══════╪═══════╪═══════╡
    │ 1    ┆ 1    ┆ a     ┆ a     │
    │ 2    ┆ 2    ┆  b    ┆  b    │
    │ 3    ┆ 3    ┆   c   ┆   c   │
    │ 4    ┆ 4    ┆ d     ┆ d     │
    │ 5    ┆ 5    ┆ e     ┆ e     │
    └──────┴──────┴───────┴───────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4  │
    │ ---  ┆ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ str  ┆ str  ┆ str   │
    ╞══════╪══════╪══════╪═══════╡
    │ 1    ┆ 1    ┆ a    ┆ a     │
    │ 2    ┆ 2    ┆ b    ┆  b    │
    │ 3    ┆ 3    ┆ c    ┆   c   │
    │ 4    ┆ 4    ┆ d    ┆ d     │
    │ 5    ┆ 5    ┆ e    ┆ e     │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = (),
        missing_policy: str = "raise",
    ) -> None:
        self._columns = tuple(columns) if columns is not None else None
        self._exclude_columns = exclude_columns

        check_column_missing_policy(missing_policy)
        self._missing_policy = missing_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "exclude_columns": self._exclude_columns,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit(self, frame: pl.DataFrame) -> None:
        self._check_input_columns(frame)
        self._fit(frame)

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_input_columns(frame)
        return self._transform(frame)

    def find_columns(self, frame: pl.DataFrame) -> tuple[str, ...]:
        r"""Find the columns to transform.

        Args:
            frame: The input DataFrame. Sometimes the columns to
                transform are found by analyzing the input
                DataFrame.

        Returns:
            The columns to transform.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.transformer import StripChars
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["a ", " b", "  c  ", "d", "e"],
        ...         "col4": ["a ", " b", "  c  ", "d", "e"],
        ...     }
        ... )
        >>> transformer = StripChars(columns=["col2", "col3"])
        >>> transformer.find_columns(frame)
        ('col2', 'col3')
        >>> transformer = StripChars()
        >>> transformer.find_columns(frame)
        ('col1', 'col2', 'col3', 'col4')

        ```
        """
        cols = list(frame.columns if self._columns is None else self._columns)
        [cols.remove(col) for col in self._exclude_columns if col in cols]
        return tuple(cols)

    def find_common_columns(self, frame: pl.DataFrame) -> tuple[str, ...]:
        r"""Find the common columns between the DataFrame columns and the
        input columns.

        Args:
            frame: The input DataFrame. Sometimes the columns to
                transform are found by analyzing the input
                DataFrame.

        Returns:
            The common columns.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.transformer import StripChars
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["a ", " b", "  c  ", "d", "e"],
        ...         "col4": ["a ", " b", "  c  ", "d", "e"],
        ...     }
        ... )
        >>> transformer = StripChars(columns=["col2", "col3", "col5"])
        >>> transformer.find_common_columns(frame)
        ('col2', 'col3')
        >>> transformer = StripChars()
        >>> transformer.find_common_columns(frame)
        ('col1', 'col2', 'col3', 'col4')

        ```
        """
        return find_common_columns(frame, self.find_columns(frame))

    def find_missing_columns(self, frame: pl.DataFrame) -> tuple[str, ...]:
        r"""Find the missing columns.

        Args:
            frame: The input DataFrame. Sometimes the columns to
                transform are found by analyzing the input
                DataFrame.

        Returns:
            The missing columns.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.transformer import StripChars
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["a ", " b", "  c  ", "d", "e"],
        ...         "col4": ["a ", " b", "  c  ", "d", "e"],
        ...     }
        ... )
        >>> transformer = StripChars(columns=["col2", "col3", "col5"])
        >>> transformer.find_missing_columns(frame)
        ('col5',)
        >>> transformer = StripChars()
        >>> transformer.find_missing_columns(frame)
        ()

        ```
        """
        return find_missing_columns(frame, self.find_columns(frame))

    def _check_input_columns(self, frame: pl.DataFrame) -> None:
        r"""Check if some input columns are missing.

        Args:
            frame: The input DataFrame to check.
        """
        check_missing_columns(
            frame_or_cols=frame,
            columns=self.find_columns(frame),
            missing_policy=self._missing_policy,
        )

    @abstractmethod
    def _fit(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Fit to the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.
        """


class BaseInNOut1Transformer(BaseInNTransformer):
    r"""Define a base class to implement ``polars.DataFrame``
    transformers that generate a single output column by using multiple
    input columns.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns.
        out_col: The output column.
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

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import ConcatColumns
    >>> transformer = ConcatColumns(columns=["col1", "col2", "col3"], out_col="col")
    >>> transformer
    ConcatColumnsTransformer(columns=('col1', 'col2', 'col3'), out_col='col', exclude_columns=(), exist_policy='raise', missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [11, 12, 13, 14, 15],
    ...         "col2": [21, 22, 23, 24, 25],
    ...         "col3": [31, 32, 33, 34, 35],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 11   ┆ 21   ┆ 31   ┆ a    │
    │ 12   ┆ 22   ┆ 32   ┆ b    │
    │ 13   ┆ 23   ┆ 33   ┆ c    │
    │ 14   ┆ 24   ┆ 34   ┆ d    │
    │ 15   ┆ 25   ┆ 35   ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.fit_transform(frame)
    >>> out
    shape: (5, 5)
    ┌──────┬──────┬──────┬──────┬──────────────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 ┆ col          │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---          │
    │ i64  ┆ i64  ┆ i64  ┆ str  ┆ list[i64]    │
    ╞══════╪══════╪══════╪══════╪══════════════╡
    │ 11   ┆ 21   ┆ 31   ┆ a    ┆ [11, 21, 31] │
    │ 12   ┆ 22   ┆ 32   ┆ b    ┆ [12, 22, 32] │
    │ 13   ┆ 23   ┆ 33   ┆ c    ┆ [13, 23, 33] │
    │ 14   ┆ 24   ┆ 34   ┆ d    ┆ [14, 24, 34] │
    │ 15   ┆ 25   ┆ 35   ┆ e    ┆ [15, 25, 35] │
    └──────┴──────┴──────┴──────┴──────────────┘


    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        out_col: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(
            columns=columns, exclude_columns=exclude_columns, missing_policy=missing_policy
        )
        self._out_col = out_col

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "columns": self._columns,
                "out_col": self._out_col,
                "exclude_columns": self._exclude_columns,
                "exist_policy": self._exist_policy,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit(self, frame: pl.DataFrame) -> None:
        self._check_input_columns(frame)
        self._fit(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_input_columns(frame)
        self._check_output_column(frame)
        return self._transform(frame)

    def _check_output_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input DataFrame to check.
        """
        check_existing_column(frame, column=self._out_col, exist_policy=self._exist_policy)

    @abstractmethod
    def _fit(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Fit to the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.
        """
