r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_clickhouse_connect",
    "check_pyarrow",
    "check_sklearn",
    "check_tqdm",
    "clickhouse_connect_available",
    "is_clickhouse_connect_available",
    "is_pyarrow_available",
    "is_sklearn_available",
    "is_tqdm_available",
    "pyarrow_available",
    "sklearn_available",
    "tqdm_available",
]

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from coola.utils.imports import decorator_package_available, package_available

if TYPE_CHECKING:
    from collections.abc import Callable


##############################
#     clickhouse_connect     #
##############################


@lru_cache
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
    return package_available("clickhouse_connect")


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
            "'clickhouse_connect' package is required but not installed. "
            "You can install 'clickhouse_connect' package with the command:\n\n"
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


####################
#     colorlog     #
####################


def is_colorlog_available() -> bool:
    r"""Indicate if the ``colorlog`` package is installed or not.

    Returns:
        ``True`` if ``colorlog`` is available otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import is_colorlog_available
    >>> is_colorlog_available()

    ```
    """
    return package_available("colorlog")


def check_colorlog() -> None:
    r"""Check if the ``colorlog`` package is installed.

    Raises:
        RuntimeError: if the ``colorlog`` package is not
            installed.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import check_colorlog
    >>> check_colorlog()

    ```
    """
    if not is_colorlog_available():
        msg = (
            "'colorlog' package is required but not installed. "
            "You can install 'colorlog' package with the command:\n\n"
            "pip install colorlog\n"
        )
        raise RuntimeError(msg)


def colorlog_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``colorlog``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``colorlog`` package is
            installed, otherwise ``None``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import colorlog_available
    >>> @colorlog_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_colorlog_available)


###################
#     pyarrow     #
###################


@lru_cache
def is_pyarrow_available() -> bool:
    r"""Indicate if the ``pyarrow`` package is installed or not.

    Returns:
        ``True`` if ``pyarrow`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import is_pyarrow_available
    >>> is_pyarrow_available()

    ```
    """
    return package_available("pyarrow")


def check_pyarrow() -> None:
    r"""Check if the ``pyarrow`` package is installed.

    Raises:
        RuntimeError: if the ``pyarrow`` package is not installed.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import check_pyarrow
    >>> check_pyarrow()

    ```
    """
    if not is_pyarrow_available():
        msg = (
            "'pyarrow' package is required but not installed. "
            "You can install 'pyarrow' package with the command:\n\n"
            "pip install pyarrow\n"
        )
        raise RuntimeError(msg)


def pyarrow_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``pyarrow``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``pyarrow`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import pyarrow_available
    >>> @pyarrow_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_pyarrow_available)


###################
#     sklearn     #
###################


@lru_cache
def is_sklearn_available() -> bool:
    r"""Indicate if the ``sklearn`` package is installed or not.

    Returns:
        ``True`` if ``sklearn`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import is_sklearn_available
    >>> is_sklearn_available()

    ```
    """
    return package_available("sklearn")


def check_sklearn() -> None:
    r"""Check if the ``sklearn`` package is installed.

    Raises:
        RuntimeError: if the ``sklearn`` package is not installed.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import check_sklearn
    >>> check_sklearn()

    ```
    """
    if not is_sklearn_available():
        msg = (
            "'sklearn' package is required but not installed. "
            "You can install 'sklearn' package with the command:\n\n"
            "pip install scikit-learn\n"
        )
        raise RuntimeError(msg)


def sklearn_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``sklearn``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``sklearn`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from grizz.utils.imports import sklearn_available
    >>> @sklearn_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_sklearn_available)


################
#     tqdm     #
################


@lru_cache
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
    return package_available("tqdm")


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
            "'tqdm' package is required but not installed. "
            "You can install 'tqdm' package with the command:\n\n"
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
