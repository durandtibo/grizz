r"""Contain utility functions to ingest DataFrames."""

from __future__ import annotations

__all__ = ["check_dataframe_file"]


from typing import TYPE_CHECKING

from grizz.exceptions import DataFrameNotFoundError

if TYPE_CHECKING:
    from pathlib import Path


def check_dataframe_file(path: Path) -> None:
    r"""Check if a DataFrame file exists or not.

    Raises:
        DataFrameNotFoundError: if the DataFrame file does not exist.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from grizz.ingestor.utils import check_dataframe_file
    >>> check_dataframe_file(Path("/path/to/frame.csv"))  # doctest: +SKIP

    ```
    """
    if not path.is_file():
        msg = f"DataFrame file does not exist: {path}"
        raise DataFrameNotFoundError(msg)
