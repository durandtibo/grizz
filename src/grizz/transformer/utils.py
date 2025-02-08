r"""Contain transformer utility functions."""

from __future__ import annotations

__all__ = ["get_classname", "message_skip_fit"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grizz.transformer.base import BaseTransformer


def get_classname(transformer: BaseTransformer) -> str:
    r"""Get the class name from the transformer object.

    Args:
        transformer: The transformer object.

    Returns:
        The transformer class name.

    Example usage:

    ```pycon

    >>> from grizz.transformer import FillNan
    >>> from grizz.transformer.utils import get_classname
    >>> get_classname(FillNan(columns=["col1", "col4"], prefix="", suffix="_out"))
    FillNanTransformer

    ```
    """
    return str(transformer.__class__.__qualname__)


def message_skip_fit(classname: str) -> str:
    r"""Generate the message to indicate the call to `fit` method is
    skipped.

    Args:
        classname: The class name of the transformer.

    Returns:
        The generated message.

    Example usage:

    ```pycon

    >>> from grizz.transformer.utils import message_skip_fit
    >>> message_skip_fit("FillNanTransformer")
    Skipping 'FillNanTransformer.fit' as there are no parameters available to fit

    ```
    """
    return f"Skipping '{classname}.fit' as there are no parameters available to fit"
