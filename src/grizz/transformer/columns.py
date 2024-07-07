r"""Contain ``polars.DataFrame`` transformers to process string
values."""

from __future__ import annotations

__all__ = ["BaseColumnsTransformer"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from grizz.transformer.base import BaseTransformer
from grizz.utils.imports import is_tqdm_available

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

if is_tqdm_available():
    from tqdm import tqdm
else:  # pragma: no cover
    from grizz.utils.noop import tqdm

logger = logging.getLogger(__name__)


class BaseColumnsTransformer(BaseTransformer):
    r"""Define a base class to implement transformers that apply the same
    transformation on multiple columns.

    Args:
        columns: The columns to prepare. If ``None``, it processes all
            the columns of type string.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import StripChars
    >>> transformer = StripChars(columns=["col2", "col3"])
    >>> transformer
    StripCharsTransformer(columns=('col2', 'col3'), ignore_missing=False)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a ", " b", "  c  ", "d", "e"],
    ...         "col4": ["a ", " b", "  c  ", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬───────┬───────┐
    │ col1 ┆ col2 ┆ col3  ┆ col4  │
    │ ---  ┆ ---  ┆ ---   ┆ ---   │
    │ i64  ┆ str  ┆ str   ┆ str   │
    ╞══════╪══════╪═══════╪═══════╡
    │ 1    ┆ 1    ┆ a     ┆ a     │
    │ 2    ┆ 2    ┆  b    ┆  b    │
    │ 3    ┆ 3    ┆   c   ┆   c   │
    │ 4    ┆ 4    ┆ d     ┆ d     │
    │ 5    ┆ 5    ┆ e     ┆ e     │
    └──────┴──────┴───────┴───────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬───────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4  │
    │ ---  ┆ ---  ┆ ---  ┆ ---   │
    │ i64  ┆ str  ┆ str  ┆ str   │
    ╞══════╪══════╪══════╪═══════╡
    │ 1    ┆ 1    ┆ a    ┆ a     │
    │ 2    ┆ 2    ┆ b    ┆  b    │
    │ 3    ┆ 3    ┆ c    ┆   c   │
    │ 4    ┆ 4    ┆ d    ┆ d     │
    │ 5    ┆ 5    ┆ e    ┆ e     │
    └──────┴──────┴──────┴───────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        ignore_missing: bool = False,
    ) -> None:
        self._columns = tuple(columns) if columns is not None else None
        self._ignore_missing = bool(ignore_missing)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        columns = self._columns
        if columns is None:
            columns = tuple(frame.columns)
        for col in tqdm(columns, desc=self._get_progressbar_message()):
            if col not in frame:
                if self._ignore_missing:
                    logger.warning(
                        f"skipping transformation for column {col} because the column is missing"
                    )
                else:
                    msg = f"column {col} is not in the DataFrame (columns:{sorted(frame.columns)})"
                    raise RuntimeError(msg)
            else:
                frame = self._transform(frame=frame, column=col)
        return frame

    @abstractmethod
    def _get_progressbar_message(self) -> str:
        r"""Return the message to show in the progress bar.

        Returns:
            The message.
        """

    @abstractmethod
    def _transform(self, frame: pl.DataFrame, column: str) -> pl.DataFrame:
        r"""Transform the data in the given column.

        Args:
            frame: The DataFrame to transform.
            column: The column to transform.

        Returns:
            The transformed DataFrame.
        """
