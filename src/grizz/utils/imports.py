r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_clickhouse_connect",
    "check_tqdm",
    "clickhouse_connect_available",
    "is_clickhouse_connect_available",
    "is_tqdm_available",
    "tqdm_available",
]

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from coola.utils.imports import decorator_package_available

if TYPE_CHECKING:
    from collections.abc import Callable


##############################
#     clickhouse_connect     #
##############################


def is_clickhouse_connect_available() -> bool:
    r"""Indicate if the ``clickhouse_connect`` package is installed or
    not.

    Returns:
        ``True`` if ``clickhouse_connect`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import is_clickhouse_connect_available
    >>> is_clickhouse_connect_available()

    ```
    """
    return find_spec("clickhouse_connect") is not None


def check_clickhouse_connect() -> None:
    r"""Check if the ``clickhouse_connect`` package is installed.

    Raises:
        RuntimeError: if the ``clickhouse_connect`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import check_clickhouse_connect
    >>> check_clickhouse_connect()

    ```
    """
    if not is_clickhouse_connect_available():
        msg = (
            "`clickhouse_connect` package is required but not installed. "
            "You can install `clickhouse_connect` package with the command:\n\n"
            "pip install clickhouse-connect\n"
        )
        raise RuntimeError(msg)


def clickhouse_connect_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if
    ``clickhouse_connect`` package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``clickhouse_connect`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import clickhouse_connect_available
    >>> @clickhouse_connect_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_clickhouse_connect_available)


################
#     tqdm     #
################


def is_tqdm_available() -> bool:
    r"""Indicate if the ``tqdm`` package is installed or not.

    Returns:
        ``True`` if ``tqdm`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import is_tqdm_available
    >>> is_tqdm_available()

    ```
    """
    return find_spec("tqdm") is not None


def check_tqdm() -> None:
    r"""Check if the ``tqdm`` package is installed.

    Raises:
        RuntimeError: if the ``tqdm`` package is not installed.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import check_tqdm
    >>> check_tqdm()

    ```
    """
    if not is_tqdm_available():
        msg = (
            "`tqdm` package is required but not installed. "
            "You can install `tqdm` package with the command:\n\n"
            "pip install tqdm\n"
        )
        raise RuntimeError(msg)


def tqdm_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``tqdm``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``tqdm`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import tqdm_available
    >>> @tqdm_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_tqdm_available)
