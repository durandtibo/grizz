r"""Contain ``polars.DataFrame`` transformers to process string
values."""

from __future__ import annotations

__all__ = ["BaseColumnsTransformer", "check_missing_columns"]

import logging
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning
from grizz.transformer.base import BaseTransformer
from grizz.utils.column import find_common_columns, find_missing_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


logger = logging.getLogger(__name__)


class BaseColumnsTransformer(BaseTransformer):
    r"""Define a base class to implement transformers that apply the same
    transformation on multiple columns.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns of type string.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import StripChars
    >>> transformer = StripChars(columns=["col2", "col3"])
    >>> transformer
    StripCharsTransformer(columns=('col2', 'col3'), ignore_missing=False)
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
        ignore_missing: bool = False,
    ) -> None:
        self._columns = tuple(columns) if columns is not None else None
        self._ignore_missing = bool(ignore_missing)

    def fit(self, frame: pl.DataFrame) -> None:
        self._pre_fit(frame)
        self._fit(frame)

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._pre_transform(frame)
        self._check_missing_columns(frame)
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
        if self._columns is None:
            return tuple(frame.columns)
        return self._columns

    def find_common_columns(self, frame: pl.DataFrame) -> tuple[str, ...]:
        r"""Find the common columns.

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

    def _check_missing_columns(self, frame: pl.DataFrame) -> None:
        r"""Check if some columns are missing.

        Args:
            frame: The input DataFrame to check.
        """
        check_missing_columns(
            frame_or_cols=frame, columns=self.find_columns(frame), missing_ok=self._ignore_missing
        )

    def _pre_fit(self, frame: pl.DataFrame) -> None:
        r"""Log information about the transformation fit.

        Args:
            frame: The DataFrame to fit.
        """

    def _fit(self, frame: pl.DataFrame) -> None:
        r"""Fit the transformer to data.

        Args:
            frame: The DataFrame to fit.
        """

    @abstractmethod
    def _pre_transform(self, frame: pl.DataFrame) -> None:
        r"""Log information about the transformation.

        Args:
            frame: The DataFrame to transform.
        """

    @abstractmethod
    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the given column.

        Args:
            frame: The DataFrame to transform.

        Returns:
            The transformed DataFrame.
        """


def check_missing_columns(
    frame_or_cols: pl.DataFrame | Sequence, columns: Sequence, missing_ok: bool = False
) -> None:
    r"""Check if some columns are missing.

    Args:
        frame_or_cols: The DataFrame or its columns.
        columns: The columns to check.
        missing_ok: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.

    Raises:
        RuntimeError: if at least one column is missing and ``missing_ok=False``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer.columns import check_missing_columns
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> check_missing_columns(frame, ["col1", "col5"], missing_ok=True)

    ```
    """
    missing_cols = find_missing_columns(frame_or_cols=frame_or_cols, columns=columns)
    if not missing_cols:
        return
    if not missing_ok:
        msg = f"{len(missing_cols):,} columns are missing in the DataFrame: {missing_cols}"
        raise ColumnNotFoundError(msg)
    msg = (
        f"{len(missing_cols):,} columns are missing in the DataFrame and will be ignored: "
        f"{missing_cols}"
    )
    warnings.warn(msg, ColumnNotFoundWarning, stacklevel=2)
