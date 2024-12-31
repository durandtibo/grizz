r"""Contain utility functions to sort values from multiple types."""

from __future__ import annotations

__all__ = ["mixed_typed_sort"]

from collections import defaultdict
from typing import TYPE_CHECKING

from grizz.utils.nan import sortnan

if TYPE_CHECKING:
    from collections.abc import Iterable


def mixed_typed_sort(iterable: Iterable, /, *, reverse: bool = False) -> list:
    r"""Return a new list containing all items from the iterable sorted
    in ascending order.

    This function is an extension of the built-in ``sorted`` function
    that works on a list with multiple types. There is no global order
    for all types, so the items are sorted only by type. For example,
    if a list has string and float values, the string values are
    sorted together and the float values are sorted together.
    Each type must implement the python sorting interface.
    The types are sorted by alphabetical order, so in the previous
    example, the float values are before the string values in the
    sorted output list. This function uses ``sortnan`` to sort
    numerical values, so it is possible to sort a list with NaNs.

    Args:
        iterable: The data to sort.
        reverse: If set to ``True``, then the list elements are sorted
            as if each comparison were reversed.

    Returns:
        The sorted data.

    Example usage:

    ```pycon

    >>> from grizz.utils.sorting import mixed_typed_sort
    >>> x = [1, "c", "a", "b", 4, -2]
    >>> mixed_typed_sort(x)
    [-2, 1, 4, 'a', 'b', 'c']
    >>> mixed_typed_sort(x, reverse=True)
    [4, 1, -2, 'c', 'b', 'a']

    ```
    """
    typed_data = defaultdict(list)
    for v in iterable:
        t = type(v)
        if t in {float, int, bool}:  # group the numeric types
            t = float
        typed_data[t].append(v)
    output = []
    # There is no global order defined between the types so we use the string representations
    # to sort the types.
    typed_data = dict(sorted(typed_data.items(), key=lambda x: x[0].__qualname__))
    for typ, values in typed_data.items():
        sort_fn = sortnan if typ is float else sorted
        output.extend(sort_fn(values, reverse=reverse))
    return output
