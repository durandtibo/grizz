r"""Contain utility functions to ingest DataFrames."""

from __future__ import annotations

__all__ = ["check_data_file"]


from typing import TYPE_CHECKING

from grizz.exceptions import DataNotFoundError

if TYPE_CHECKING:
    from pathlib import Path


def check_data_file(path: Path) -> None:
    r"""Check if the file containing data exists or not.

    Raises:
        DataNotFoundError: if the file does not exist.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from grizz.ingestor.utils import check_data_file
    >>> check_data_file(Path("/path/to/frame.csv"))  # doctest: +SKIP

    ```
    """
    if not path.is_file():
        msg = f"Data file does not exist: {path}"
        raise DataNotFoundError(msg)
