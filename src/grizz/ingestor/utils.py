r"""Contain utility functions to ingest DataFrames."""

from __future__ import annotations

__all__ = ["check_dataframe_path"]


from typing import TYPE_CHECKING

from grizz.exceptions import DataFrameNotFoundError

if TYPE_CHECKING:
    from pathlib import Path


def check_dataframe_path(path: Path) -> None:
    r"""Check if a DataFrame path exists or not.

    Raises:
        DataFrameNotFoundError: if the DataFrame path does not exist.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from grizz.ingestor.utils import check_dataframe_path
    >>> check_dataframe_path(Path("/path/to/frame.csv"))  # doctest: +SKIP

    ```
    """
    if not path.exists():
        msg = f"DataFrame path does not exist: {path}"
        raise DataFrameNotFoundError(msg)
