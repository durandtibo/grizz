r"""Contain the base class to implement a ``polars.DataFrame``
transformer."""

from __future__ import annotations

__all__ = [
    "BaseDataFrameTransformer",
    "is_dataframe_transformer_config",
    "setup_dataframe_transformer",
]

import logging
from abc import ABC
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class BaseDataFrameTransformer(ABC, metaclass=AbstractFactory):
    r"""Define the base class to transform a ``polars.DataFrame``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer.dataframe import ToNumeric
    >>> transformer = ToNumeric(columns=["col1", "col3"])
    >>> transformer
    ToNumericDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame.dtypes
    col1     int64
    col2    object
    col3    object
    col4    object
    dtype: object
    >>> out = transformer.transform(frame)
    >>> out.dtypes
    col1     int64
    col2    object
    col3     int64
    col4    object
    dtype: object

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
        >>> from grizz.transformer.dataframe import ToNumeric
        >>> transformer = ToNumeric(columns=["col1", "col3"])
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["1", "2", "3", "4", "5"],
        ...         "col4": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> out = transformer.transform(frame)
        >>> out.dtypes
        col1     int64
        col2    object
        col3     int64
        col4    object
        dtype: object

        ```
        """


def is_dataframe_transformer_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDataFrameTransformer``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseDataFrameTransformer`` object.

    Example usage:

    ```pycon

    >>> from grizz.transformer.dataframe import is_dataframe_transformer_config
    >>> is_dataframe_transformer_config(
    ...     {"_target_": "grizz.transformer.dataframe.ToNumeric", "columns": ["col1", "col3"]}
    ... )
    True

    ```
    """
    return is_object_config(config, BaseDataFrameTransformer)


def setup_dataframe_transformer(
    transformer: BaseDataFrameTransformer | dict,
) -> BaseDataFrameTransformer:
    r"""Set up a ``polars.DataFrame`` transformer.

    The transformer is instantiated from its configuration
    by using the ``BaseDataFrameTransformer`` factory function.

    Args:
        transformer: Specifies a ``polars.DataFrame`` transformer or
            its configuration.

    Returns:
        An instantiated transformer.

    Example usage:

    ```pycon

    >>> from grizz.transformer.dataframe import setup_dataframe_transformer
    >>> transformer = setup_dataframe_transformer(
    ...     {"_target_": "grizz.transformer.dataframe.ToNumeric", "columns": ["col1", "col3"]}
    ... )
    >>> transformer
    ToNumericDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False)

    ```
    """
    if isinstance(transformer, dict):
        logger.info("Initializing a DataFrame transformer from its configuration... ")
        transformer = BaseDataFrameTransformer.factory(**transformer)
    if not isinstance(transformer, BaseDataFrameTransformer):
        logger.warning(
            f"transformer is not a `BaseDataFrameTransformer` (received: {type(transformer)})"
        )
    return transformer
