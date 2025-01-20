r"""Contain the base class to implement a ``polars.DataFrame``
transformer."""

from __future__ import annotations

__all__ = [
    "BaseTransformer",
    "is_transformer_config",
    "setup_transformer",
]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.equality.comparators import BaseEqualityComparator
from coola.equality.handlers import EqualNanHandler, SameObjectHandler, SameTypeHandler
from coola.equality.testers import EqualityTester
from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    import polars as pl
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class BaseTransformer(ABC, metaclass=AbstractFactory):
    r"""Define the base class to transform a ``polars.DataFrame``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import InplaceCast
    >>> transformer = InplaceCast(columns=["col1", "col3"], dtype=pl.Int32)
    >>> transformer
    InplaceCastTransformer(columns=('col1', 'col3'), exclude_columns=(), missing_policy='raise', dtype=Int32)
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
    >>> out = transformer.transform(frame)
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

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The other object to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.transformer import InplaceCast
        >>> obj1 = InplaceCast(columns=["col1", "col3"], dtype=pl.Int32)
        >>> obj2 = InplaceCast(columns=["col1", "col3"], dtype=pl.Int32)
        >>> obj3 = InplaceCast(columns=["col2", "col3"], dtype=pl.Float32)
        >>> obj1.equal(obj2)
        True
        >>> obj1.equal(obj3)
        False

        ```
        """

    @abstractmethod
    def fit(self, frame: pl.DataFrame) -> None:
        r"""Fit to the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to fit.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.transformer import InplaceCast
        >>> transformer = InplaceCast(columns=["col1", "col3"], dtype=pl.Int32)
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
        >>> transformer.fit(frame)
        >>> out = transformer.transform(frame)
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

    @abstractmethod
    def fit_transform(self, frame: pl.DataFrame) -> None:
        r"""Fit to the data, then transform it.

        Args:
            frame: The ``polars.DataFrame`` to fit.

        Returns:
            The transformed DataFrame.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.transformer import InplaceCast
        >>> transformer = InplaceCast(columns=["col1", "col3"], dtype=pl.Int32)
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
        >>> out = transformer.fit_transform(frame)
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

    @abstractmethod
    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.transformer import InplaceCast
        >>> transformer = InplaceCast(columns=["col1", "col3"], dtype=pl.Int32)
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
        >>> out = transformer.transform(frame)
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


def is_transformer_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTransformer``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseTransformer`` object.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import is_transformer_config
    >>> is_transformer_config(
    ...     {
    ...         "_target_": "grizz.transformer.InplaceCast",
    ...         "columns": ("col1", "col3"),
    ...         "dtype": pl.Int32,
    ...     }
    ... )
    True

    ```
    """
    return is_object_config(config, BaseTransformer)


def setup_transformer(
    transformer: BaseTransformer | dict,
) -> BaseTransformer:
    r"""Set up a ``polars.DataFrame`` transformer.

    The transformer is instantiated from its configuration
    by using the ``BaseTransformer`` factory function.

    Args:
        transformer: Specifies a ``polars.DataFrame`` transformer or
            its configuration.

    Returns:
        An instantiated transformer.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import setup_transformer
    >>> transformer = setup_transformer(
    ...     {
    ...         "_target_": "grizz.transformer.InplaceCast",
    ...         "columns": ("col1", "col3"),
    ...         "dtype": pl.Int32,
    ...     }
    ... )
    >>> transformer
    InplaceCastTransformer(columns=('col1', 'col3'), exclude_columns=(), missing_policy='raise', dtype=Int32)

    ```
    """
    if isinstance(transformer, dict):
        logger.info("Initializing a DataFrame transformer from its configuration... ")
        transformer = BaseTransformer.factory(**transformer)
    if not isinstance(transformer, BaseTransformer):
        logger.warning(f"transformer is not a `BaseTransformer` (received: {type(transformer)})")
    return transformer


class TransformerEqualityComparator(BaseEqualityComparator[BaseTransformer]):
    r"""Implement an equality comparator for ``BaseTransformer``
    objects."""

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualNanHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TransformerEqualityComparator:
        return self.__class__()

    def equal(self, actual: BaseTransformer, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


if not EqualityTester.has_comparator(BaseTransformer):  # pragma: no cover
    EqualityTester.add_comparator(BaseTransformer, TransformerEqualityComparator())
