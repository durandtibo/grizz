from __future__ import annotations

__all__ = []


import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from grizz.processor.base import BaseProcessor

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class MinMaxScalerProcessor(BaseProcessor):

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._scaler.fit_transform()


if __name__ == "__main__":
    scaler = MinMaxScaler()
    X = np.random.randn(100, 3)
    out = scaler.fit(X)
    attrs = {
        "min_": scaler.min_,
        "scale_": scaler.scale_,
        "data_range_": scaler.data_range_,
        "data_min_": scaler.data_min_,
        "data_max_": scaler.data_max_,
        "n_features_in_": scaler.n_features_in_,
        "n_samples_seen_": scaler.n_samples_seen_,
    }
