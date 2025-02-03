r"""Contain the base class to implement a ``polars.LazyFrame``
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
    r"""Define the base class to transform a ``polars.LazyFrame``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.lazy.transformer import DropNullRow
    >>> transformer = DropNullRow()
    >>> transformer
    DropNullRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.LazyFrame(
    ...     {
    ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
    ...         "col2": [1, None, 3, None, None],
    ...         "col3": [None, None, None, None, None],
    ...     }
    ... )
    >>> frame.collect()
    shape: (5, 3)
    ┌────────────┬──────┬──────┐
    │ col1       ┆ col2 ┆ col3 │
    │ ---        ┆ ---  ┆ ---  │
    │ str        ┆ i64  ┆ null │
    ╞════════════╪══════╪══════╡
    │ 2020-1-1   ┆ 1    ┆ null │
    │ 2020-1-2   ┆ null ┆ null │
    │ 2020-1-31  ┆ 3    ┆ null │
    │ 2020-12-31 ┆ null ┆ null │
    │ null       ┆ null ┆ null │
    └────────────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out.collect()
    shape: (4, 3)
    ┌────────────┬──────┬──────┐
    │ col1       ┆ col2 ┆ col3 │
    │ ---        ┆ ---  ┆ ---  │
    │ str        ┆ i64  ┆ null │
    ╞════════════╪══════╪══════╡
    │ 2020-1-1   ┆ 1    ┆ null │
    │ 2020-1-2   ┆ null ┆ null │
    │ 2020-1-31  ┆ 3    ┆ null │
    │ 2020-12-31 ┆ null ┆ null │
    └────────────┴──────┴──────┘

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
        >>> from grizz.lazy.transformer import DropNullRow
        >>> obj1 = DropNullRow(columns=["col1", "col3"])
        >>> obj2 = DropNullRow(columns=["col1", "col3"])
        >>> obj3 = DropNullRow(columns=["col2", "col3"])
        >>> obj1.equal(obj2)
        True
        >>> obj1.equal(obj3)
        False

        ```
        """

    @abstractmethod
    def fit(self, frame: pl.LazyFrame) -> None:
        r"""Fit to the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to fit.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.lazy.transformer import DropNullRow
        >>> transformer = DropNullRow()
        >>> transformer
        DropNullRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
        >>> frame = pl.LazyFrame(
        ...     {
        ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
        ...         "col2": [1, None, 3, None, None],
        ...         "col3": [None, None, None, None, None],
        ...     }
        ... )
        >>> frame.collect()
        shape: (5, 3)
        ┌────────────┬──────┬──────┐
        │ col1       ┆ col2 ┆ col3 │
        │ ---        ┆ ---  ┆ ---  │
        │ str        ┆ i64  ┆ null │
        ╞════════════╪══════╪══════╡
        │ 2020-1-1   ┆ 1    ┆ null │
        │ 2020-1-2   ┆ null ┆ null │
        │ 2020-1-31  ┆ 3    ┆ null │
        │ 2020-12-31 ┆ null ┆ null │
        │ null       ┆ null ┆ null │
        └────────────┴──────┴──────┘
        >>> transformer.fit(frame)

        ```
        """

    @abstractmethod
    def fit_transform(self, frame: pl.LazyFrame) -> None:
        r"""Fit to the data, then transform it.

        Args:
            frame: The ``polars.LazyFrame`` to fit.

        Returns:
            The transformed LazyFrame.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.lazy.transformer import DropNullRow
        >>> transformer = DropNullRow()
        >>> transformer
        DropNullRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
        >>> frame = pl.LazyFrame(
        ...     {
        ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
        ...         "col2": [1, None, 3, None, None],
        ...         "col3": [None, None, None, None, None],
        ...     }
        ... )
        >>> frame.collect()
        shape: (5, 3)
        ┌────────────┬──────┬──────┐
        │ col1       ┆ col2 ┆ col3 │
        │ ---        ┆ ---  ┆ ---  │
        │ str        ┆ i64  ┆ null │
        ╞════════════╪══════╪══════╡
        │ 2020-1-1   ┆ 1    ┆ null │
        │ 2020-1-2   ┆ null ┆ null │
        │ 2020-1-31  ┆ 3    ┆ null │
        │ 2020-12-31 ┆ null ┆ null │
        │ null       ┆ null ┆ null │
        └────────────┴──────┴──────┘
        >>> out = transformer.fit_transform(frame)
        >>> out.collect()
        shape: (4, 3)
        ┌────────────┬──────┬──────┐
        │ col1       ┆ col2 ┆ col3 │
        │ ---        ┆ ---  ┆ ---  │
        │ str        ┆ i64  ┆ null │
        ╞════════════╪══════╪══════╡
        │ 2020-1-1   ┆ 1    ┆ null │
        │ 2020-1-2   ┆ null ┆ null │
        │ 2020-1-31  ┆ 3    ┆ null │
        │ 2020-12-31 ┆ null ┆ null │
        └────────────┴──────┴──────┘

        ```
        """

    @abstractmethod
    def transform(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        r"""Transform the data in the ``polars.LazyFrame``.

        Args:
            frame: The ``polars.LazyFrame`` to transform.

        Returns:
            The transformed LazyFrame.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from grizz.lazy.transformer import DropNullRow
        >>> transformer = DropNullRow()
        >>> transformer
        DropNullRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')
        >>> frame = pl.LazyFrame(
        ...     {
        ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
        ...         "col2": [1, None, 3, None, None],
        ...         "col3": [None, None, None, None, None],
        ...     }
        ... )
        >>> frame.collect()
        shape: (5, 3)
        ┌────────────┬──────┬──────┐
        │ col1       ┆ col2 ┆ col3 │
        │ ---        ┆ ---  ┆ ---  │
        │ str        ┆ i64  ┆ null │
        ╞════════════╪══════╪══════╡
        │ 2020-1-1   ┆ 1    ┆ null │
        │ 2020-1-2   ┆ null ┆ null │
        │ 2020-1-31  ┆ 3    ┆ null │
        │ 2020-12-31 ┆ null ┆ null │
        │ null       ┆ null ┆ null │
        └────────────┴──────┴──────┘
        >>> out = transformer.transform(frame)
        >>> out.collect()
        shape: (4, 3)
        ┌────────────┬──────┬──────┐
        │ col1       ┆ col2 ┆ col3 │
        │ ---        ┆ ---  ┆ ---  │
        │ str        ┆ i64  ┆ null │
        ╞════════════╪══════╪══════╡
        │ 2020-1-1   ┆ 1    ┆ null │
        │ 2020-1-2   ┆ null ┆ null │
        │ 2020-1-31  ┆ 3    ┆ null │
        │ 2020-12-31 ┆ null ┆ null │
        └────────────┴──────┴──────┘

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

    >>> from grizz.lazy.transformer import is_transformer_config
    >>> is_transformer_config({"_target_": "grizz.lazy.transformer.DropNullRow"})
    True

    ```
    """
    return is_object_config(config, BaseTransformer)


def setup_transformer(
    transformer: BaseTransformer | dict,
) -> BaseTransformer:
    r"""Set up a ``polars.LazyFrame`` transformer.

    The transformer is instantiated from its configuration
    by using the ``BaseTransformer`` factory function.

    Args:
        transformer: Specifies a ``polars.LazyFrame`` transformer or
            its configuration.

    Returns:
        An instantiated transformer.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.lazy.transformer import setup_transformer
    >>> transformer = setup_transformer(
    ...     {
    ...         "_target_": "grizz.lazy.transformer.DropNullRow",
    ...     }
    ... )
    >>> transformer
    DropNullRowTransformer(columns=None, exclude_columns=(), missing_policy='raise')

    ```
    """
    if isinstance(transformer, dict):
        logger.info("Initializing a LazyFrame transformer from its configuration... ")
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
