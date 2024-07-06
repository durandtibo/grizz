r"""Contain utility functions to format strings."""

from __future__ import annotations

__all__ = ["human_byte"]


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
