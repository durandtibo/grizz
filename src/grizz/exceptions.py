r"""Contain custom exceptions."""

from __future__ import annotations

__all__ = [
    "ColumnExistsError",
    "ColumnExistsWarning",
    "ColumnNotFoundError",
    "ColumnNotFoundWarning",
    "DataNotFoundError",
    "TransformerNotFittedError",
]


class ColumnExistsError(RuntimeError):
    r"""Raised when trying to create a column which already exists."""


class ColumnExistsWarning(RuntimeWarning):
    r"""Raised when trying to create a column which already exists."""


class ColumnNotFoundError(RuntimeError):
    r"""Raised when a column is requested but does not exist."""


class ColumnNotFoundWarning(RuntimeWarning):
    r"""Raised when a column is requested but does not exist."""


class DataNotFoundError(RuntimeError):
    r"""Raised when a DataFrame is requested but does not exist."""


class TransformerNotFittedError(RuntimeError):
    r"""Raised when a transformer is used before to fitting."""
