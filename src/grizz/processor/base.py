r"""Contain the base class to implement a ``polars.DataFrame``
processor."""

from __future__ import annotations

__all__ = [
    "BaseProcessor",
    "is_processor_config",
    "setup_processor",
]

import logging
from abc import ABC
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class BaseProcessor(ABC, metaclass=AbstractFactory):
    r"""Define the base class to process a ``polars.DataFrame``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.processor import Cast
    >>> processor = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    >>> processor
    CastProcessor(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = processor.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i32  ┆ str  ┆ i32  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.processor import Cast
        >>> processor = Cast(columns=["col1", "col3"], dtype=pl.Int32)
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["1", "2", "3", "4", "5"],
        ...         "col4": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> frame
        shape: (5, 4)
        ┌──────┬──────┬──────┬──────┐
        │ col1 ┆ col2 ┆ col3 ┆ col4 │
        │ ---  ┆ ---  ┆ ---  ┆ ---  │
        │ i64  ┆ str  ┆ str  ┆ str  │
        ╞══════╪══════╪══════╪══════╡
        │ 1    ┆ 1    ┆ 1    ┆ a    │
        │ 2    ┆ 2    ┆ 2    ┆ b    │
        │ 3    ┆ 3    ┆ 3    ┆ c    │
        │ 4    ┆ 4    ┆ 4    ┆ d    │
        │ 5    ┆ 5    ┆ 5    ┆ e    │
        └──────┴──────┴──────┴──────┘
        >>> out = processor.transform(frame)
        >>> out
        shape: (5, 4)
        ┌──────┬──────┬──────┬──────┐
        │ col1 ┆ col2 ┆ col3 ┆ col4 │
        │ ---  ┆ ---  ┆ ---  ┆ ---  │
        │ i32  ┆ str  ┆ i32  ┆ str  │
        ╞══════╪══════╪══════╪══════╡
        │ 1    ┆ 1    ┆ 1    ┆ a    │
        │ 2    ┆ 2    ┆ 2    ┆ b    │
        │ 3    ┆ 3    ┆ 3    ┆ c    │
        │ 4    ┆ 4    ┆ 4    ┆ d    │
        │ 5    ┆ 5    ┆ 5    ┆ e    │
        └──────┴──────┴──────┴──────┘

        ```
        """


def is_processor_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseProcessor``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseProcessor`` object.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.processor import is_processor_config
    >>> is_processor_config(
    ...     {
    ...         "_target_": "grizz.processor.Cast",
    ...         "columns": ("col1", "col3"),
    ...         "dtype": pl.Int32,
    ...     }
    ... )
    True

    ```
    """
    return is_object_config(config, BaseProcessor)


def setup_processor(
    processor: BaseProcessor | dict,
) -> BaseProcessor:
    r"""Set up a ``polars.DataFrame`` processor.

    The processor is instantiated from its configuration
    by using the ``BaseProcessor`` factory function.

    Args:
        processor: A ``polars.DataFrame`` processor or its
            configuration.

    Returns:
        An instantiated processor.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.processor import setup_processor
    >>> processor = setup_processor(
    ...     {
    ...         "_target_": "grizz.processor.Cast",
    ...         "columns": ("col1", "col3"),
    ...         "dtype": pl.Int32,
    ...     }
    ... )
    >>> processor
    CastProcessor(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)

    ```
    """
    if isinstance(processor, dict):
        logger.info("Initializing a DataFrame processor from its configuration... ")
        processor = BaseProcessor.factory(**processor)
    if not isinstance(processor, BaseProcessor):
        logger.warning(f"processor is not a `BaseProcessor` (received: {type(processor)})")
    return processor
