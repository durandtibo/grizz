r"""Contain utility functions to transform data with NaNs."""

from __future__ import annotations

__all__ = ["remove_nan", "sortnan"]

import math
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

T = TypeVar("T", bound=Sequence)


def remove_nan(data: T) -> T:
    r"""Remove the NaN values from the input sequence.

    Args:
        data: The input sequence.

    Returns:
        The input sequence without NaN values.

    Example usage:

    ```pycon

    >>> from grizz.utils.nan import remove_nan
    >>> data = [float("nan"), float("-inf"), -2, 1.2]
    >>> remove_nan(data)
    [-inf, -2, 1.2]

    ```
    """
    return type(data)([x for x in data if not isinstance(x, (float, int)) or not math.isnan(x)])


def sortnan(iterable: Iterable[bool | float], /, *, reverse: bool = False) -> list[bool | float]:
    r"""Sort a sequence of numeric values with NaN.

    This function is an extension of the built-in ``sorted`` function.
    It sees NaN values as equivalent to -infinity when the values are
    sorted.

    Args:
        iterable: The numeric values to sort.
        reverse: If set to ``True``, then the list elements are sorted
            as if each comparison were reversed.

    Returns:
        The sorted list.

    Example usage:

    ```pycon

    >>> from grizz.utils.nan import sortnan
    >>> x = [4, float("nan"), 2, 1.2, 7.9, -2]
    >>> sorted(x)
    [4, nan, -2, 1.2, 2, 7.9]
    >>> sortnan(x)
    [nan, -2, 1.2, 2, 4, 7.9]
    >>> sortnan(x, reverse=True)
    [7.9, 4, 2, 1.2, -2, nan]

    ```
    """
    return sorted(iterable, key=lambda x: LowNaN() if math.isnan(x) else x, reverse=reverse)


class LowNaN(float):
    r"""Implement a NaN representation that is always lower than other
    numbers.

    This class is designed to be used to compare numbers with NaN values
    and should not be used in other cases.

    https://docs.python.org/3/library/functions.html#sorted
    """

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, (float, int)):
            msg = (
                f"'>=' not supported between instances of 'float' and '{type(other).__qualname__}'"
            )
            raise TypeError(msg)
        return False

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, (float, int)):
            msg = f"'>' not supported between instances of 'float' and '{type(other).__qualname__}'"
            raise TypeError(msg)
        return False

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, (float, int)):
            msg = (
                f"'<=' not supported between instances of 'float' and '{type(other).__qualname__}'"
            )
            raise TypeError(msg)
        return True

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, (float, int)):
            msg = f"'<' not supported between instances of 'float' and '{type(other).__qualname__}'"
            raise TypeError(msg)
        return True
