r"""Contain utility functions to compute the co-occurrence matrix."""

from __future__ import annotations

__all__ = ["compute_pairwise_cooccurrence"]

from unittest.mock import Mock

import polars as pl
from coola.utils import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


def compute_pairwise_cooccurrence(frame: pl.DataFrame, ignore_self: bool = False) -> np.ndarray:
    r"""Compute the pairwise column co-occurrence.

    Args:
        frame: The input DataFrame. The column values are expected to
            be 0/1 or true/false.
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.

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
    >>> compute_pairwise_cooccurrence(frame, ignore_self=True)
    array([[0, 2, 1],
           [2, 0, 1],
           [1, 1, 0]])

    ```
    """
    check_numpy()
    if frame.shape[1] == 0:
        return np.zeros((0, 0), dtype=int)
    data = frame.cast(pl.Boolean).fill_null(False).to_numpy().astype(int)
    co = data.transpose().dot(data)
    if ignore_self:
        np.fill_diagonal(co, 0)
    return co
