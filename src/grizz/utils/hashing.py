r"""Contain utility functions to compute hash of objects."""

from __future__ import annotations

__all__ = ["str_to_sha256"]

import hashlib


def str_to_sha256(string: str) -> str:
    r"""Generate the SHA-256 hash of a string.

    Args:
        string: The string to hash.

    Returns:
        The SHA-256 hash.

    Example usage:

    ```pycon

    >>> from grizz.utils.hashing import str_to_sha256
    >>> str_to_sha256("bears are funny")
    c97afc5c7f1b598c9f68dc2d6e323b2dd2eaaa31d3a07c98059de6079cbd30e0

    ```
    """
    return hashlib.sha256(str(string).encode("utf-8")).hexdigest()
