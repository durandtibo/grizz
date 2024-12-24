r"""Contain custom exceptions."""

from __future__ import annotations

__all__ = [
    "ColumnExistsError",
    "ColumnExistsWarning",
    "ColumnNotFoundError",
    "ColumnNotFoundWarning",
    "DataFrameNotFoundError",
]


class ColumnExistsError(RuntimeError):
    r"""Raised when trying to create a column which already exists."""


class ColumnExistsWarning(RuntimeWarning):
    r"""Raised when trying to create a column which already exists."""


class ColumnNotFoundError(RuntimeError):
    r"""Raised when a column is requested but does not exist."""


class ColumnNotFoundWarning(RuntimeWarning):
    r"""Raised when a column is requested but does not exist."""


class DataFrameNotFoundError(RuntimeError):
    r"""Raised when a DataFrame is requested but does not exist."""
