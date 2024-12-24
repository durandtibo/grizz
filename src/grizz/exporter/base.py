r"""Contain the base class to implement a DataFrame exporter."""

from __future__ import annotations

__all__ = ["BaseExporter", "is_exporter_config", "setup_exporter"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class BaseExporter(ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a DataFrame exporter.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.exporter import ParquetExporter
    >>> exporter = ParquetExporter(path="/path/to/frame.parquet")
    >>> exporter
    ParquetExporter(path=/path/to/frame.parquet)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> exporter.export(frame)  # doctest: +SKIP

    ```
    """

    @abstractmethod
    def export(self, frame: pl.DataFrame) -> None:
        r"""Export a DataFrame.

        Args:
            frame: The DataFrame to export.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.exporter import ParquetExporter
        >>> exporter = ParquetExporter(path="/path/to/frame.parquet")
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> exporter.export(frame)  # doctest: +SKIP

        ```
        """


def is_exporter_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseExporter``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseExporter`` object.

    Example usage:

    ```pycon

    >>> from grizz.exporter import is_exporter_config
    >>> is_exporter_config(
    ...     {"_target_": "grizz.exporter.ParquetExporter", "path": "/path/to/data.parquet"}
    ... )
    True

    ```
    """
    return is_object_config(config, BaseExporter)


def setup_exporter(
    exporter: BaseExporter | dict,
) -> BaseExporter:
    r"""Set up an exporter.

    The exporter is instantiated from its configuration
    by using the ``BaseExporter`` factory function.

    Args:
        exporter: A exporter or its configuration.

    Returns:
        An instantiated exporter.

    Example usage:

    ```pycon

    >>> from grizz.exporter import setup_exporter
    >>> exporter = setup_exporter(
    ...     {"_target_": "grizz.exporter.ParquetExporter", "path": "/path/to/data.parquet"}
    ... )
    >>> exporter
    ParquetExporter(path=/path/to/data.parquet)

    ```
    """
    if isinstance(exporter, dict):
        logger.info("Initializing an exporter from its configuration... ")
        exporter = BaseExporter.factory(**exporter)
    if not isinstance(exporter, BaseExporter):
        logger.warning(f"exporter is not a `BaseExporter` (received: {type(exporter)})")
    return exporter
