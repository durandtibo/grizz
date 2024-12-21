from __future__ import annotations

from unittest.mock import patch

import pytest

from grizz.utils.imports import (
    check_clickhouse_connect,
    check_pyarrow,
    check_sklearn,
    check_tqdm,
    clickhouse_connect_available,
    is_clickhouse_connect_available,
    is_pyarrow_available,
    is_sklearn_available,
    is_tqdm_available,
    pyarrow_available,
    sklearn_available,
    tqdm_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


##############################
#     clickhouse_connect     #
##############################


def test_check_clickhouse_connect_with_package() -> None:
    with patch("grizz.utils.imports.is_clickhouse_connect_available", lambda: True):
        check_clickhouse_connect()


def test_check_clickhouse_connect_without_package() -> None:
    with (
        patch("grizz.utils.imports.is_clickhouse_connect_available", lambda: False),
        pytest.raises(
            RuntimeError, match="'clickhouse_connect' package is required but not installed."
        ),
    ):
        check_clickhouse_connect()


def test_is_clickhouse_connect_available() -> None:
    assert isinstance(is_clickhouse_connect_available(), bool)


def test_clickhouse_connect_available_with_package() -> None:
    with patch("grizz.utils.imports.is_clickhouse_connect_available", lambda: True):
        fn = clickhouse_connect_available(my_function)
        assert fn(2) == 44


def test_clickhouse_connect_available_without_package() -> None:
    with patch("grizz.utils.imports.is_clickhouse_connect_available", lambda: False):
        fn = clickhouse_connect_available(my_function)
        assert fn(2) is None


def test_clickhouse_connect_available_decorator_with_package() -> None:
    with patch("grizz.utils.imports.is_clickhouse_connect_available", lambda: True):

        @clickhouse_connect_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_clickhouse_connect_available_decorator_without_package() -> None:
    with patch("grizz.utils.imports.is_clickhouse_connect_available", lambda: False):

        @clickhouse_connect_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


###################
#     pyarrow     #
###################


def test_check_pyarrow_with_package() -> None:
    with patch("grizz.utils.imports.is_pyarrow_available", lambda: True):
        check_pyarrow()


def test_check_pyarrow_without_package() -> None:
    with (
        patch("grizz.utils.imports.is_pyarrow_available", lambda: False),
        pytest.raises(RuntimeError, match="'pyarrow' package is required but not installed."),
    ):
        check_pyarrow()


def test_is_pyarrow_available() -> None:
    assert isinstance(is_pyarrow_available(), bool)


def test_pyarrow_available_with_package() -> None:
    with patch("grizz.utils.imports.is_pyarrow_available", lambda: True):
        fn = pyarrow_available(my_function)
        assert fn(2) == 44


def test_pyarrow_available_without_package() -> None:
    with patch("grizz.utils.imports.is_pyarrow_available", lambda: False):
        fn = pyarrow_available(my_function)
        assert fn(2) is None


def test_pyarrow_available_decorator_with_package() -> None:
    with patch("grizz.utils.imports.is_pyarrow_available", lambda: True):

        @pyarrow_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pyarrow_available_decorator_without_package() -> None:
    with patch("grizz.utils.imports.is_pyarrow_available", lambda: False):

        @pyarrow_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


###################
#     sklearn     #
###################


def test_check_sklearn_with_package() -> None:
    with patch("grizz.utils.imports.is_sklearn_available", lambda: True):
        check_sklearn()


def test_check_sklearn_without_package() -> None:
    with (
        patch("grizz.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        check_sklearn()


def test_is_sklearn_available() -> None:
    assert isinstance(is_sklearn_available(), bool)


def test_sklearn_available_with_package() -> None:
    with patch("grizz.utils.imports.is_sklearn_available", lambda: True):
        fn = sklearn_available(my_function)
        assert fn(2) == 44


def test_sklearn_available_without_package() -> None:
    with patch("grizz.utils.imports.is_sklearn_available", lambda: False):
        fn = sklearn_available(my_function)
        assert fn(2) is None


def test_sklearn_available_decorator_with_package() -> None:
    with patch("grizz.utils.imports.is_sklearn_available", lambda: True):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_sklearn_available_decorator_without_package() -> None:
    with patch("grizz.utils.imports.is_sklearn_available", lambda: False):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


################
#     tqdm     #
################


def test_check_tqdm_with_package() -> None:
    with patch("grizz.utils.imports.is_tqdm_available", lambda: True):
        check_tqdm()


def test_check_tqdm_without_package() -> None:
    with (
        patch("grizz.utils.imports.is_tqdm_available", lambda: False),
        pytest.raises(RuntimeError, match="'tqdm' package is required but not installed."),
    ):
        check_tqdm()


def test_is_tqdm_available() -> None:
    assert isinstance(is_tqdm_available(), bool)


def test_tqdm_available_with_package() -> None:
    with patch("grizz.utils.imports.is_tqdm_available", lambda: True):
        fn = tqdm_available(my_function)
        assert fn(2) == 44


def test_tqdm_available_without_package() -> None:
    with patch("grizz.utils.imports.is_tqdm_available", lambda: False):
        fn = tqdm_available(my_function)
        assert fn(2) is None


def test_tqdm_available_decorator_with_package() -> None:
    with patch("grizz.utils.imports.is_tqdm_available", lambda: True):

        @tqdm_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_tqdm_available_decorator_without_package() -> None:
    with patch("grizz.utils.imports.is_tqdm_available", lambda: False):

        @tqdm_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
