r"""Contain DataFrame columns utility functions."""

from __future__ import annotations

__all__ = ["find_missing_columns"]


from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


def find_missing_columns(
    frame_or_cols: pl.DataFrame | Sequence, columns: Sequence[str]
) -> list[str]:
    r"""Find the given columns that are not in the DataFrame (or its
    columns).

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
    ['col4']

    ```
    """
    cols = set(frame_or_cols.columns if isinstance(frame_or_cols, pl.DataFrame) else frame_or_cols)
    columns = set(columns)
    return sorted(columns.difference(cols).intersection(columns))
