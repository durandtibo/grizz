from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.transformer import CloseColumns


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "col3": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
    )


#############################################
#     Tests for CloseColumnsTransformer     #
#############################################


def test_close_columns_transformer_repr() -> None:
    assert str(CloseColumns(actual="col1", expected="col2", out_col="out")).startswith(
        "CloseColumnsTransformer("
    )


def test_close_columns_transformer_str() -> None:
    assert str(CloseColumns(actual="col1", expected="col2", out_col="out")).startswith(
        "CloseColumnsTransformer("
    )


def test_close_columns_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = CloseColumns(actual="col1", expected="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'CloseColumnsTransformer.fit' as there are no parameters available to fit"
    )


def test_close_columns_transformer_fit_missing_policy_ignore_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(
        actual="col", expected="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_close_columns_transformer_fit_missing_policy_ignore_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(
        actual="col1", expected="missing", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_close_columns_transformer_fit_missing_policy_raise_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col", expected="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_close_columns_transformer_fit_missing_policy_raise_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col1", expected="missing", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_close_columns_transformer_fit_missing_policy_warn_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col", expected="col2", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignore"
    ):
        transformer.fit(dataframe)


def test_close_columns_transformer_fit_missing_policy_warn_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(
        actual="col1", expected="missing", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'missing' is missing in the DataFrame and will be ignore",
    ):
        transformer.fit(dataframe)


def test_close_columns_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = CloseColumns(actual="col1", expected="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [False, False, True, False, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_close_columns_transformer_transform_float(dataframe: pl.DataFrame) -> None:
    transformer = CloseColumns(actual="col1", expected="col2", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [False, False, True, False, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_close_columns_transformer_transform_int() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [5, 4, 3, 2, 1],
            "col3": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
    )
    transformer = CloseColumns(actual="col1", expected="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [False, False, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_close_columns_transformer_transform_empty() -> None:
    frame = pl.DataFrame(
        {"col1": [], "col2": []},
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = CloseColumns(actual="col1", expected="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [], "col2": [], "out": []},
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_close_columns_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(
        actual="col1", expected="col2", out_col="col3", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": [False, False, True, False, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Boolean},
        ),
    )


def test_close_columns_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col1", expected="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_close_columns_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col1", expected="col2", out_col="col3", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col3' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": [False, False, True, False, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.Boolean},
        ),
    )


def test_close_columns_transformer_transform_missing_policy_ignore_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(
        actual="col", expected="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
        ),
    )


def test_close_columns_transformer_transform_missing_policy_ignore_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(
        actual="col1", expected="missing", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
        ),
    )


def test_close_columns_transformer_transform_missing_policy_raise_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col", expected="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_close_columns_transformer_transform_missing_policy_raise_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col1", expected="missing", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_close_columns_transformer_transform_missing_policy_warn_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(actual="col", expected="col2", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignore"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
        ),
    )


def test_close_columns_transformer_transform_missing_policy_warn_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = CloseColumns(
        actual="col1", expected="missing", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'missing' is missing in the DataFrame and will be ignore",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
        ),
    )
