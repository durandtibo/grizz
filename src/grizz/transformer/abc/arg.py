r"""Contain a base class to implement transformers with custom
arguments."""

from __future__ import annotations

__all__ = ["BaseArgTransformer"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils.format import repr_mapping_line, str_mapping_line
from iden.utils.time import timeblock

from grizz.exceptions import TransformerNotFittedError
from grizz.transformer.base import BaseTransformer
from grizz.utils.format import str_dataframe_diff

if TYPE_CHECKING:
    import polars as pl

    from grizz.types import Self

logger = logging.getLogger(__name__)


class BaseArgTransformer(BaseTransformer):
    r"""Define a base class to implement transformers with custom
    arguments."""

    def __init__(self, requires_fit: bool) -> None:
        self._requires_fit = requires_fit
        self._is_fitted = False

    def __repr__(self) -> str:
        args = repr_mapping_line(self.get_args())
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_mapping_line(self.get_args())
        return f"{self.__class__.__qualname__}({args})"

    def check_is_fitted(self) -> None:
        r"""Indicate if the transformer is fitted or not."""
        if self._requires_fit and not self._is_fitted:
            msg = "This transformer instance is not fitted yet"
            raise TransformerNotFittedError(msg)

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    def fit(self, frame: pl.DataFrame) -> Self:
        with timeblock(f"{self.__class__.__qualname__}.fit - " + "time: {time}"):
            self._fit(frame)
        self._is_fitted = True
        return self

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return self.fit(frame).transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.check_is_fitted()
        with timeblock(f"{self.__class__.__qualname__}.transform - " + "time: {time}"):
            out = self._transform(frame)
        logger.info(str_dataframe_diff(orig=frame, final=out))
        return out

    @abstractmethod
    def get_args(self) -> dict:
        r"""Get the arguments of the transformer.

        Returns:
            The arguments of the transformer.
        """

    @abstractmethod
    def _fit(self, frame: pl.DataFrame) -> None:
        r"""Fit to the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to fit.
        """

    @abstractmethod
    def _transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        r"""Transform the data in the ``polars.DataFrame``.

        Args:
            frame: The ``polars.DataFrame`` to transform.

        Returns:
            The transformed DataFrame.
        """
