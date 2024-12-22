r"""Contain a base class to implement ``polars.DataFrame`` transformers
that transform a single input column and generate an output column."""

from __future__ import annotations

__all__ = ["BaseColumnTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils.format import repr_mapping_line

from grizz.transformer.base import BaseTransformer
from grizz.utils.column import (
    check_column_exist_policy,
    check_column_missing_policy,
    check_existing_columns,
    check_missing_columns,
)

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class BaseColumnTransformer(BaseTransformer):
    r"""Define a base class to implement ``polars.DataFrame``
    transformers that transform a single input column and generate an
    output column.

    Args:
        in_col: The input column name.
        out_col: The output column name.
        exist_policy: The policy on how to handle existing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column already exist.
            If ``'warn'``, a warning is raised if at least one column
            already exist and the existing columns are overwritten.
            If ``'ignore'``, the existing columns are overwritten and
            no warning message is shown.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message is shown.
    """

    def __init__(
        self,
        in_col: str,
        out_col: str,
        exist_policy: str = "raise",
        missing_policy: str = "raise",
    ) -> None:
        self._in_col = in_col
        self._out_col = out_col

        check_column_exist_policy(exist_policy)
        self._exist_policy = exist_policy
        check_column_missing_policy(missing_policy)
        self._missing_policy = missing_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "in_col": self._in_col,
                "out_col": self._out_col,
                "exist_policy": self._exist_policy,
                "missing_policy": self._missing_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def _check_input_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the input column is missing.

        Args:
            frame: The input DataFrame to check.
        """
        check_missing_columns(frame, columns=[self._in_col], missing_policy=self._missing_policy)

    def _check_output_column(self, frame: pl.DataFrame) -> None:
        r"""Check if the output column already exists.

        Args:
            frame: The input DataFrame to check.
        """
        check_existing_columns(frame, columns=[self._out_col], exist_policy=self._exist_policy)
