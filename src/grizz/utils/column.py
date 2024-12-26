r"""Contain DataFrame columns utility functions."""

from __future__ import annotations

__all__ = [
    "check_column_exist_policy",
    "check_column_missing_policy",
    "check_existing_column",
    "check_existing_columns",
    "check_missing_column",
    "check_missing_columns",
    "find_common_columns",
    "find_missing_columns",
]

import warnings
from typing import TYPE_CHECKING

import polars as pl

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def check_column_exist_policy(exist_policy: str) -> None:
    r"""Check the policy on how to handle existing columns.

    Args:
        exist_policy: The policy on how to handle existing columns.

    Raises:
        ValueError: if ``exist_policy`` is not ``'ignore'``,
            ``'warn'``, or ``'raise'``.

    Example usage:

    ```pycon

    >>> from grizz.utils.column import check_column_exist_policy
    >>> check_column_exist_policy("ignore")

    ```
    """
    if exist_policy not in {"ignore", "warn", "raise"}:
        msg = (
            f"Incorrect 'exist_policy': {exist_policy}. The valid values are: "
            f"'ignore', 'raise', 'warn'"
        )
        raise ValueError(msg)


def check_column_missing_policy(missing_policy: str) -> None:
    r"""Check the policy on how to handle missing columns.

    Args:
        missing_policy: The policy on how to handle missing columns.

    Raises:
        ValueError: if ``missing_policy`` is not ``'ignore'``,
            ``'warn'``, or ``'raise'``.

    Example usage:

    ```pycon

    >>> from grizz.utils.column import check_column_missing_policy
    >>> check_column_missing_policy("ignore")

    ```
    """
    if missing_policy not in {"ignore", "warn", "raise"}:
        msg = (
            f"Incorrect 'missing_policy': {missing_policy}. The valid values are: "
            f"'ignore', 'raise', 'warn'"
        )
        raise ValueError(msg)


def check_existing_column(
    frame_or_cols: pl.DataFrame | Sequence, column: str, exist_policy: str = "raise"
) -> None:
    r"""Check if a column already exists.

    Args:
        frame_or_cols: The DataFrame or its columns.
        column: The column to check.
        exist_policy: The policy on how to handle existing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised the column already exists. If ``'warn'``,
            a warning is raised if the column already exists and the
            existing column is overwritten. If ``'ignore'``,
            the existing column is overwritten and no warning message
            appears.

    Raises:
        ColumnExistsError: if at least one column already exists and
            ``exist_policy='raise'``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.column import check_existing_column
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> check_existing_column(frame, "col1", exist_policy="warn")

    ```
    """
    check_column_exist_policy(exist_policy)
    exist = column in frame_or_cols
    if not exist:
        return
    msg = f"column {column!r} already exists in the DataFrame"
    if exist_policy == "raise":
        raise ColumnExistsError(msg)
    if exist_policy == "warn":
        warnings.warn(msg + " and will be overwritten", ColumnExistsWarning, stacklevel=2)


def check_existing_columns(
    frame_or_cols: pl.DataFrame | Sequence, columns: Sequence, exist_policy: str = "raise"
) -> None:
    r"""Check if some columns already exist.

    Args:
        frame_or_cols: The DataFrame or its columns.
        columns: The columns to check.
        exist_policy: The policy on how to handle existing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column already exist.
            If ``'warn'``, a warning is raised if at least one column
            already exist and the existing columns are overwritten.
            If ``'ignore'``, the existing columns are overwritten and
            no warning message appears.

    Raises:
        ColumnExistsError: if at least one column already exists and
            ``exist_policy='raise'``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.column import check_existing_columns
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> check_existing_columns(frame, ["col1", "col5"], exist_policy="warn")

    ```
    """
    check_column_exist_policy(exist_policy)
    existing_cols = find_common_columns(frame_or_cols=frame_or_cols, columns=columns)
    if not existing_cols:
        return
    m = "column already exists" if len(existing_cols) == 1 else "columns already exist"
    if exist_policy == "raise":
        msg = f"{len(existing_cols):,} {m} in the DataFrame: {existing_cols}"
        raise ColumnExistsError(msg)
    if exist_policy == "warn":
        msg = (
            f"{len(existing_cols):,} {m} in the DataFrame "
            f"and will be overwritten: {existing_cols}"
        )
        warnings.warn(msg, ColumnExistsWarning, stacklevel=2)


def check_missing_column(
    frame_or_cols: pl.DataFrame | Sequence, column: str, missing_policy: str = "raise"
) -> None:
    r"""Check if a column is missing.

    Args:
        frame_or_cols: The DataFrame or its columns.
        column: The column to check.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if the column is missing. If ``'warn'``,
            a warning is raised if the column is missing and the
            missing columns are ignored. If ``'ignore'``, the missing
            column is ignored and no warning message appears.

    Raises:
        ColumnNotFoundError: if the column is missing and
            ``missing_policy='raise'``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.column import check_missing_column
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> check_missing_column(frame, "col1", missing_policy="warn")

    ```
    """
    check_column_missing_policy(missing_policy)
    exist = column in frame_or_cols
    if exist:
        return
    msg = f"column {column!r} is missing in the DataFrame"
    if missing_policy == "raise":
        raise ColumnNotFoundError(msg)
    if missing_policy == "warn":
        warnings.warn(msg + " and will be ignored", ColumnNotFoundWarning, stacklevel=2)


def check_missing_columns(
    frame_or_cols: pl.DataFrame | Sequence, columns: Sequence, missing_policy: str = "raise"
) -> None:
    r"""Check if some columns are missing.

    Args:
        frame_or_cols: The DataFrame or its columns.
        columns: The columns to check.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message appears.

    Raises:
        ColumnNotFoundError: if at least one column is missing and
            ``missing_policy='raise'``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.column import check_missing_columns
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> check_missing_columns(frame, ["col1", "col5"], missing_policy="warn")

    ```
    """
    check_column_missing_policy(missing_policy)
    missing_cols = find_missing_columns(frame_or_cols=frame_or_cols, columns=columns)
    if not missing_cols:
        return
    m = "column is" if len(missing_cols) == 1 else "columns are"
    if missing_policy == "raise":
        msg = f"{len(missing_cols):,} {m} missing in the DataFrame: {missing_cols}"
        raise ColumnNotFoundError(msg)
    if missing_policy == "warn":
        msg = (
            f"{len(missing_cols):,} {m} missing in the DataFrame and "
            f"will be ignored: {missing_cols}"
        )
        warnings.warn(msg, ColumnNotFoundWarning, stacklevel=2)


def find_common_columns(
    frame_or_cols: pl.DataFrame | Sequence, columns: Sequence[str]
) -> tuple[str, ...]:
    r"""Find the common columns that are both in the DataFrame and the
    given columns.

    Args:
        frame_or_cols: The DataFrame or its columns.
        columns: The columns to check.

    Returns:
        The columns i.e. the columns that are both in
            ``columns`` and ``frame_or_cols``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.column import find_common_columns
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> cols = find_common_columns(frame, columns=["col1", "col2", "col3", "col4"])
    >>> cols
    ('col1', 'col2', 'col3')

    ```
    """
    cols = set(frame_or_cols.columns if isinstance(frame_or_cols, pl.DataFrame) else frame_or_cols)
    columns = set(columns)
    return tuple(sorted(columns.intersection(cols)))


def find_missing_columns(
    frame_or_cols: pl.DataFrame | Sequence, columns: Sequence[str]
) -> tuple[str, ...]:
    r"""Find the columns that are in the given columns but not in the
    DataFrame.

    Args:
        frame_or_cols: The DataFrame or its columns.
        columns: The columns to check.

    Returns:
        The list of missing columns i.e. the columns that are in
            ``columns`` but not in ``frame_or_cols``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.column import find_missing_columns
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> cols = find_missing_columns(frame, columns=["col1", "col2", "col3", "col4"])
    >>> cols
    ('col4',)

    ```
    """
    cols = set(frame_or_cols.columns if isinstance(frame_or_cols, pl.DataFrame) else frame_or_cols)
    columns = set(columns)
    return tuple(sorted(columns.difference(cols).intersection(columns)))
