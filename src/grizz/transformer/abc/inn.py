r"""Contain a base class to implement transformers that transform
multiple columns of a DataFrame."""

from __future__ import annotations

__all__ = ["BaseInNTransformer"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from grizz.exceptions import TransformerNotFittedError
from grizz.transformer.abc.arg import BaseArgTransformer
from grizz.utils.column import (
    check_column_missing_policy,
    check_missing_columns,
    find_common_columns,
    find_missing_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


logger = logging.getLogger(__name__)


class BaseInNTransformer(BaseArgTransformer):
    r"""Define a base class to implement transformers that transform
    multiple columns of a DataFrame.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If a column is not found, it will be ignored
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
    >>> from grizz.transformer import DropNullRow
    >>> transformer = DropNullRow()
    >>> transformer
    DropNullRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
    ...         "col2": [1, None, 3, None, None],
    ...         "col3": [None, None, None, None, None],
    ...     }
    ... )
    >>> frame
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
    >>> out
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
        super().__init__()
        self._columns = tuple(columns) if columns is not None else None
        self._exclude_columns = exclude_columns

        check_column_missing_policy(missing_policy)
        self._missing_policy = missing_policy

    def get_args(self) -> dict:
        return {
            "columns": self._columns,
            "exclude_columns": self._exclude_columns,
            "missing_policy": self._missing_policy,
        }

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
        >>> from grizz.transformer import DropNullRow
        >>> frame = pl.DataFrame(
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
        >>> from grizz.transformer import DropNullRow
        >>> frame = pl.DataFrame(
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
        return find_common_columns(frame, self._columns)

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
        >>> from grizz.transformer import DropNullRow
        >>> frame = pl.DataFrame(
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
        return find_missing_columns(frame, self._columns)

    def get_input_columns(self) -> tuple[str, ...]:
        if self._columns is None:
            msg = (
                "Input columns are unknown. Call 'fit' to initialize the columns "
                "before to call 'get_input_columns'"
            )
            raise TransformerNotFittedError(msg)
        return self._columns

    def _check_input_columns(self, frame: pl.DataFrame) -> None:
        r"""Check if some input columns are missing.

        Args:
            frame: The input DataFrame to check.
        """
        check_missing_columns(
            frame_or_cols=frame,
            columns=self._columns,
            missing_policy=self._missing_policy,
        )

    def _fit(self, frame: pl.DataFrame) -> None:
        self._check_input_columns(frame)
        self._fit_data(frame)

    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._check_input_columns(frame)
        return self._transform_data(frame)

    @abstractmethod
    def _fit_data(self, frame: pl.DataFrame) -> None:
        r"""Fit to the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to fit.
        """

    @abstractmethod
    def _transform_data(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.
        """
