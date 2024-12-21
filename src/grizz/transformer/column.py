r"""Contain a base class to implement ``polars.DataFrame`` transformers
that assumes there's a 1-to-1 correspondence between the input column
and output column."""

from __future__ import annotations

__all__ = ["BaseOneToOneColumnTransformer"]

import logging
from abc import abstractmethod

import polars as pl
from coola.utils.format import repr_mapping_line

from grizz.transformer.base import BaseTransformer

logger = logging.getLogger(__name__)


class BaseOneToOneColumnTransformer(BaseTransformer):
    r"""Define a base class to implement ``polars.DataFrame``
    transformers that assumes there's a 1-to-1 correspondence between
    the input column and output column.

    Args:
        in_col: The input column name.
        out_col: The output column name.
        overwrite: If ``False``, raise an error if the output column
            already exists, otherwise the column is overwritten.
        ignore_missing: If ``True``, this transformer is skipped if the
            input column is missing.
    """

    def __init__(
        self,
        in_col: str,
        out_col: str,
        overwrite: bool = False,
        ignore_missing: bool = False,
    ) -> None:
        self._in_col = in_col
        self._out_col = out_col
        self._overwrite = overwrite
        self._ignore_missing = ignore_missing

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "in_col": self._in_col,
                "out_col": self._out_col,
                "overwrite": self._overwrite,
                "ignore_missing": self._ignore_missing,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit(self, frame: pl.DataFrame) -> None:
        self._pre_fit(frame)
        self._fit(frame)

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._pre_transform(frame)
        self._check_out_col(frame)
        return frame.with_columns(
            self._transform(frame.select(pl.col(self._in_col))).to_frame(self._out_col)
        )

    def _check_out_col(self, frame: pl.DataFrame) -> None:
        if self._out_col in frame and not self._overwrite:
            msg = (
                f"Output column '{self._out_col}' already exists. Use `overwrite=True` "
                f"to overwrite the output column"
            )
            raise RuntimeError(msg)

    @abstractmethod
    def _pre_fit(self, frame: pl.DataFrame) -> None:
        r"""Log information about the transformation fit.

        Args:
            frame: The DataFrame to fit.
        """

    @abstractmethod
    def _fit(self, frame: pl.DataFrame) -> None:
        r"""Fit the transformer to data.

        Args:
            frame: The DataFrame to fit.
        """

    @abstractmethod
    def _pre_transform(self, frame: pl.DataFrame) -> None:
        r"""Log information about the transformation.

        Args:
            frame: The DataFrame to transform.
        """

    @abstractmethod
    def _transform(self, frame: pl.Series) -> pl.Series:
        r"""Transform the data.

        Args:
            frame: The Series to transform.

        Returns:
            The transformed Series.
        """
