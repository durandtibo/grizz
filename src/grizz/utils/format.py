r"""Contain utility functions to format strings."""

from __future__ import annotations

__all__ = [
    "human_byte",
    "str_col_diff",
    "str_kwargs",
    "str_row_diff",
    "str_shape_diff",
]


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def human_byte(size: float, decimal: int = 2) -> str:
    r"""Return a human-readable string representation of byte sizes.

    Args:
        size: The number of bytes.
        decimal: The number of decimal digits.

    Returns:
        The human-readable string representation of byte sizes.

    Example usage:

    ```pycon

    >>> from grizz.utils.format import human_byte
    >>> human_byte(2)
    '2.00 B'
    >>> human_byte(2048)
    '2.00 KB'
    >>> human_byte(2097152)
    '2.00 MB'

    ```
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0:
            break
        if unit != "PB":
            size /= 1024.0
    return f"{size:,.{decimal}f} {unit}"


def str_kwargs(mapping: Mapping) -> str:
    r"""Return a string of the input mapping.

    This function is designed to be used in ``__repr__`` and
    ``__str__`` methods.

    Args:
        mapping: The mapping.

    Returns:
        The generated string.

    Example usage:

    ```pycon

    >>> from grizz.utils.format import str_kwargs
    >>> str_kwargs({"key1": 1})
    ', key1=1'
    >>> str_kwargs({"key1": 1, "key2": 2})
    ', key1=1, key2=2'

    ```
    """
    args = ", ".join([f"{key}={value}" for key, value in mapping.items()])
    if args:
        args = ", " + args
    return args


def str_col_diff(orig: int, final: int) -> str:
    r"""Return a string that indicates the difference of columns.

    Args:
        orig: The original number of columns.
        final: The final number of columns.

    Returns:
        The generated string with the difference of columns.

    Example usage:

    ```pycon

    >>> from grizz.utils.format import str_col_diff
    >>> str_col_diff(100, 10)
    90/100 (90.0000 %) columns have been removed
    >>> str_col_diff(100, 99)
    1/100 (1.0000 %) column has been removed

    ```
    """
    diff = abs(orig - final)
    desc = "removed" if orig >= final else "added"
    diff_pct = 100 * diff / orig if orig > 0 else float("nan")
    row = "columns have" if diff > 1 else "column has"
    return f"{diff:,}/{orig:,} ({diff_pct:.4f} %) {row} been {desc}"


def str_row_diff(orig: int, final: int) -> str:
    r"""Return a string that indicates the difference of rows.

    Args:
        orig: The original number of rows.
        final: The final number of rows.

    Returns:
        The generated string with the difference of rows.

    Example usage:

    ```pycon

    >>> from grizz.utils.format import str_row_diff
    >>> str_row_diff(100, 10)
    90/100 (90.0000 %) rows have been removed
    >>> str_row_diff(100, 99)
    1/100 (1.0000 %) row has been removed

    ```
    """
    diff = abs(orig - final)
    desc = "removed" if orig >= final else "added"
    diff_pct = 100 * diff / orig if orig > 0 else float("nan")
    row = "rows have" if diff > 1 else "row has"
    return f"{diff:,}/{orig:,} ({diff_pct:.4f} %) {row} been {desc}"


def str_shape_diff(orig: tuple[int, int], final: tuple[int, int]) -> str:
    r"""Return a string that indicates the difference of DataFrame
    shapes.

    Args:
        orig: The original shape.
        final: The final number of rows.

    Returns:
        The generated string with the difference of DataFrame shapes.

    Example usage:

    ```pycon

    >>> from grizz.utils.format import str_shape_diff
    >>> str_shape_diff(orig=(100, 10), final=(80, 8))
    DataFrame shape: (100, 10) -> (80, 8) | 20/100 (20.0000 %) rows have been removed |
    2/10 (20.0000 %) columns have been removed

    ```
    """
    msg = f"DataFrame shape: {orig} -> {final}"
    if orig[0] != final[0]:
        msg += f" | {str_row_diff(orig=orig[0], final=final[0])}"
    if orig[1] != final[1]:
        msg += f" | {str_col_diff(orig=orig[1], final=final[1])}"
    return msg
