r"""Contain utility functions to compute the co-occurrence matrix."""

from __future__ import annotations

__all__ = ["compute_pairwise_cooccurrence"]

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


def compute_pairwise_cooccurrence(frame: pl.DataFrame, remove_diagonal: bool = False) -> np.ndarray:
    r"""Compute the pairwise column co-occurrence.

    Args:
        frame: The input DataFrame. The column values are expected to
            be 0/1 or true/false.
        remove_diagonal: If ``True``, the diagonal of the co-occurrence
            matrix is set to 0.

    Returns:
        The co-occurrence matrix.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.utils.cooccurrence import compute_pairwise_cooccurrence
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> compute_pairwise_cooccurrence(frame)
    array([[3, 2, 1],
           [2, 3, 1],
           [1, 1, 3]])
    >>> compute_pairwise_cooccurrence(frame, remove_diagonal=True)
    array([[0, 2, 1],
           [2, 0, 1],
           [1, 1, 0]])

    ```
    """
    if frame.shape[1] == 0:
        return np.zeros((0, 0), dtype=int)
    data = frame.to_numpy().astype(bool).astype(int)
    co = data.transpose().dot(data)
    if remove_diagonal:
        np.fill_diagonal(co, 0)
    return co
