r"""Contain a base class to implement ``polars.LazyFrame`` transformers
that transform columns of LazyFrames."""

from __future__ import annotations

__all__ = [
    "BaseArgTransformer",
    "BaseIn1Out1Transformer",
    "BaseIn2Out1Transformer",
    "BaseInNOut1Transformer",
    "BaseInNOutNTransformer",
    "BaseInNTransformer",
]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils.format import repr_mapping_line

from grizz.lazy.transformer.base import BaseTransformer
from grizz.utils.column import (
    check_column_exist_policy,
    check_column_missing_policy,
    check_existing_column,
    check_existing_columns,
    check_missing_column,
    check_missing_columns,
    find_common_columns,
    find_missing_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

logger = logging.getLogger(__name__)


class BaseArgTransformer(BaseTransformer):
    r"""Define a base class to implement transformers with custom
    arguments."""

    def __repr__(self) -> str:
        args = repr_mapping_line(self.get_args())
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    def fit_transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        self.fit(frame)
        return self.transform(frame)

    @abstractmethod
    def get_args(self) -> dict:
        r"""Get the arguments of the transformer.

        Returns:
            The arguments of the transformer.
        """

    @abstractmethod
    def fit(self, frame: pl.LazyFrame) -> None:
        r"""Fit to the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to fit.
        """

    @abstractmethod
    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Transform the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to transform.

        Returns:
            The transformed LazyFrame.
        """


class BaseIn1Out1Transformer(BaseArgTransformer):
    r"""Define a base class to implement ``polars.LazyFrame``
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

    def fit(self, frame: pl.LazyFrame) -> None:
        self._check_input_column(frame)
        if self._in_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.fit' "
                f"because the input column {self._in_col!r} is missing"
            )
            return
        self._fit(frame)

    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        self._check_input_column(frame)
        if self._in_col not in frame:
            logger.info(
                f"Skipping '{self.__class__.__qualname__}.transform' "
                f"because the input column {self._in_col!r} is missing"
            )
            return frame
        self._check_output_column(frame)
        return self._transform(frame)

    def get_args(self) -> dict:
        return {
            "in_col": self._in_col,
            "out_col": self._out_col,
            "exist_policy": self._exist_policy,
            "missing_policy": self._missing_policy,
        }

    def _check_input_column(self, frame: pl.LazyFrame) -> None:
        r"""Check if the input column is missing.

        Args:
            frame: The input LazyFrame to check.
        """
        check_missing_column(
            frame.collect_schema().names(), column=self._in_col, missing_policy=self._missing_policy
        )

    def _check_output_column(self, frame: pl.LazyFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input LazyFrame to check.
        """
        check_existing_column(
            frame.collect_schema().names(), column=self._out_col, exist_policy=self._exist_policy
        )

    @abstractmethod
    def _fit(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Fit to the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Transform the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to transform.

        Returns:
            The transformed LazyFrame.
        """


class BaseIn2Out1Transformer(BaseArgTransformer):
    r"""Define a base class to implement ``polars.LazyFrame``
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

    def fit(self, frame: pl.LazyFrame) -> None:
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

    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
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

    def get_args(self) -> dict:
        return {
            "in1_col": self._in1_col,
            "in2_col": self._in2_col,
            "out_col": self._out_col,
            "exist_policy": self._exist_policy,
            "missing_policy": self._missing_policy,
        }

    def _check_input_columns(self, frame: pl.LazyFrame) -> None:
        r"""Check if any of the input columns is missing.

        Args:
            frame: The input LazyFrame to check.
        """
        check_missing_column(
            frame.collect_schema().names(),
            column=self._in1_col,
            missing_policy=self._missing_policy,
        )
        check_missing_column(
            frame.collect_schema().names(),
            column=self._in2_col,
            missing_policy=self._missing_policy,
        )

    def _check_output_column(self, frame: pl.LazyFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input LazyFrame to check.
        """
        check_existing_column(
            frame.collect_schema().names(), column=self._out_col, exist_policy=self._exist_policy
        )

    @abstractmethod
    def _fit(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Fit to the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Transform the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to transform.

        Returns:
            The transformed LazyFrame.
        """


class BaseInNTransformer(BaseArgTransformer):
    r"""Define a base class to implement ``polars.LazyFrame``
    transformers that transform LazyFrames by using multiple input
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
    >>> from grizz.lazy.transformer import DropNullRow
    >>> transformer = DropNullRow()
    >>> transformer
    DropNullRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.LazyFrame(
    ...     {
    ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
    ...         "col2": [1, None, 3, None, None],
    ...         "col3": [None, None, None, None, None],
    ...     }
    ... )
    >>> frame.collect()
    shape: (5, 3)
    ┌────────────┬──────┬──────┐
    │ col1       ┆ col2 ┆ col3 │
    │ ---        ┆ ---  ┆ ---  │
    │ str        ┆ i64  ┆ null │
    ╞════════════╪══════╪══════╡
    │ 2020-1-1   ┆ 1    ┆ null │
    │ 2020-1-2   ┆ null ┆ null │
    │ 2020-1-31  ┆ 3    ┆ null │
    │ 2020-12-31 ┆ null ┆ null │
    │ null       ┆ null ┆ null │
    └────────────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out.collect()
    shape: (4, 3)
    ┌────────────┬──────┬──────┐
    │ col1       ┆ col2 ┆ col3 │
    │ ---        ┆ ---  ┆ ---  │
    │ str        ┆ i64  ┆ null │
    ╞════════════╪══════╪══════╡
    │ 2020-1-1   ┆ 1    ┆ null │
    │ 2020-1-2   ┆ null ┆ null │
    │ 2020-1-31  ┆ 3    ┆ null │
    │ 2020-12-31 ┆ null ┆ null │
    └────────────┴──────┴──────┘

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

    def fit(self, frame: pl.LazyFrame) -> None:
        self._check_input_columns(frame)
        self._fit(frame)

    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        self._check_input_columns(frame)
        return self._transform(frame)

    def get_args(self) -> dict:
        return {
            "columns": self._columns,
            "exclude_columns": self._exclude_columns,
            "missing_policy": self._missing_policy,
        }

    def find_columns(self, frame: pl.LazyFrame) -> tuple[str, ...]:
        r"""Find the columns to transform.

        Args:
            frame: The input LazyFrame. Sometimes the columns to
                transform are found by analyzing the input
                LazyFrame.

        Returns:
            The columns to transform.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.lazy.transformer import DropNullRow
        >>> frame = pl.LazyFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["a ", " b", "  c  ", "d", "e"],
        ...         "col4": ["a ", " b", "  c  ", "d", "e"],
        ...     }
        ... )
        >>> transformer = DropNullRow(columns=["col2", "col3"])
        >>> transformer.find_columns(frame)
        ('col2', 'col3')
        >>> transformer = DropNullRow()
        >>> transformer.find_columns(frame)
        ('col1', 'col2', 'col3', 'col4')

        ```
        """
        cols = list(frame.collect_schema().names() if self._columns is None else self._columns)
        [cols.remove(col) for col in self._exclude_columns if col in cols]
        return tuple(cols)

    def find_common_columns(self, frame: pl.LazyFrame) -> tuple[str, ...]:
        r"""Find the common columns between the LazyFrame columns and the
        input columns.

        Args:
            frame: The input LazyFrame. Sometimes the columns to
                transform are found by analyzing the input
                LazyFrame.

        Returns:
            The common columns.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.lazy.transformer import DropNullRow
        >>> frame = pl.LazyFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["a ", " b", "  c  ", "d", "e"],
        ...         "col4": ["a ", " b", "  c  ", "d", "e"],
        ...     }
        ... )
        >>> transformer = DropNullRow(columns=["col2", "col3", "col5"])
        >>> transformer.find_common_columns(frame)
        ('col2', 'col3')
        >>> transformer = DropNullRow()
        >>> transformer.find_common_columns(frame)
        ('col1', 'col2', 'col3', 'col4')

        ```
        """
        return find_common_columns(frame.collect_schema().names(), self.find_columns(frame))

    def find_missing_columns(self, frame: pl.LazyFrame) -> tuple[str, ...]:
        r"""Find the missing columns.

        Args:
            frame: The input LazyFrame. Sometimes the columns to
                transform are found by analyzing the input
                LazyFrame.

        Returns:
            The missing columns.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.lazy.transformer import DropNullRow
        >>> frame = pl.LazyFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["a ", " b", "  c  ", "d", "e"],
        ...         "col4": ["a ", " b", "  c  ", "d", "e"],
        ...     }
        ... )
        >>> transformer = DropNullRow(columns=["col2", "col3", "col5"])
        >>> transformer.find_missing_columns(frame)
        ('col5',)
        >>> transformer = DropNullRow()
        >>> transformer.find_missing_columns(frame)
        ()

        ```
        """
        return find_missing_columns(frame.collect_schema().names(), self.find_columns(frame))

    def _check_input_columns(self, frame: pl.LazyFrame) -> None:
        r"""Check if some input columns are missing.

        Args:
            frame: The input LazyFrame to check.
        """
        check_missing_columns(
            frame_or_cols=frame.collect_schema().names(),
            columns=self.find_columns(frame),
            missing_policy=self._missing_policy,
        )

    @abstractmethod
    def _fit(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Fit to the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Transform the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to transform.

        Returns:
            The transformed LazyFrame.
        """


class BaseInNOut1Transformer(BaseInNTransformer):
    r"""Define a base class to implement ``polars.LazyFrame``
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

    def fit(self, frame: pl.LazyFrame) -> None:
        self._check_input_columns(frame)
        self._fit(frame)

    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        self._check_input_columns(frame)
        self._check_output_column(frame)
        return self._transform(frame)

    def get_args(self) -> dict:
        return {
            "columns": self._columns,
            "out_col": self._out_col,
            "exclude_columns": self._exclude_columns,
            "exist_policy": self._exist_policy,
            "missing_policy": self._missing_policy,
        }

    def _check_output_column(self, frame: pl.LazyFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input LazyFrame to check.
        """
        check_existing_column(frame, column=self._out_col, exist_policy=self._exist_policy)


class BaseInNOutNTransformer(BaseInNTransformer):
    r"""Define a base class to implement ``polars.LazyFrame``
    transformers that has N input columns and N output columns.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns.
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
    """

    def __init__(
        self,
        columns: Sequence[str] | None,
        prefix: str,
        suffix: str,
        exclude_columns: Sequence[str] = (),
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(
            columns=columns, exclude_columns=exclude_columns, missing_policy=missing_policy
        )
        self._prefix = prefix
        self._suffix = suffix

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy

    def fit(self, frame: pl.LazyFrame) -> None:
        self._check_input_columns(frame)
        self._fit(frame)

    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        self._check_input_columns(frame)
        self._check_output_column(frame)
        out = self._transform(frame)
        return frame.with_columns(out.rename(lambda col: f"{self._prefix}{col}{self._suffix}"))

    def get_args(self) -> dict:
        return {
            "columns": self._columns,
            "exclude_columns": self._exclude_columns,
            "exist_policy": self._exist_policy,
            "missing_policy": self._missing_policy,
            "prefix": self._prefix,
            "suffix": self._suffix,
        }

    def _check_output_column(self, frame: pl.LazyFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input LazyFrame to check.
        """
        check_existing_columns(
            frame.collect_schema().names(),
            columns=[f"{self._prefix}{col}{self._suffix}" for col in self.find_columns(frame)],
            exist_policy=self._exist_policy,
        )
