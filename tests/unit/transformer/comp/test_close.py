from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from grizz.exceptions import (
    ColumnExistsError,
    ColumnExistsWarning,
    ColumnNotFoundError,
    ColumnNotFoundWarning,
)
from grizz.transformer import ColumnClose


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


############################################
#     Tests for ColumnCloseTransformer     #
############################################


def test_column_close_transformer_repr() -> None:
    assert repr(ColumnClose(actual="col1", expected="col2", out_col="out")).startswith(
        "ColumnCloseTransformer("
    )


def test_column_close_transformer_str() -> None:
    assert str(ColumnClose(actual="col1", expected="col2", out_col="out")).startswith(
        "ColumnCloseTransformer("
    )


def test_column_close_transformer_equal_true() -> None:
    assert ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col2", out_col="out")
    )


def test_column_close_transformer_equal_false_actual() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col", expected="col2", out_col="out")
    )


def test_column_close_transformer_equal_false_expected() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col", out_col="out")
    )


def test_column_close_transformer_equal_false_out_col() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col2", out_col="out2")
    )


def test_column_close_transformer_equal_false_atol() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col2", out_col="out", atol=1e-5)
    )


def test_column_close_transformer_equal_false_rtol() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col2", out_col="out", rtol=1e-3)
    )


def test_column_close_transformer_equal_false_equal_nan() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col2", out_col="out", equal_nan=True)
    )


def test_column_close_transformer_equal_false_exist_policy() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col2", out_col="out", exist_policy="warn")
    )


def test_column_close_transformer_equal_false_missing_policy() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(
        ColumnClose(actual="col1", expected="col2", out_col="out", missing_policy="warn")
    )


def test_column_close_transformer_equal_false_different_type() -> None:
    assert not ColumnClose(actual="col1", expected="col2", out_col="out").equal(42)


def test_column_close_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnClose(actual="col1", expected="col2", out_col="out").get_args(),
        {
            "actual": "col1",
            "expected": "col2",
            "out_col": "out",
            "atol": 1e-8,
            "rtol": 1e-5,
            "equal_nan": False,
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_close_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnCloseTransformer.fit' as there are no parameters available to fit"
    )


def test_column_close_transformer_fit_missing_policy_ignore_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col", expected="col2", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_close_transformer_fit_missing_policy_ignore_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(
        actual="col1", expected="missing", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_close_transformer_fit_missing_policy_raise_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col", expected="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_close_transformer_fit_missing_policy_raise_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col1", expected="missing", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_close_transformer_fit_missing_policy_warn_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col", expected="col2", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignore"
    ):
        transformer.fit(dataframe)


def test_column_close_transformer_fit_missing_policy_warn_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(
        actual="col1", expected="missing", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'missing' is missing in the DataFrame and will be ignore",
    ):
        transformer.fit(dataframe)


def test_column_close_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out")
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


def test_column_close_transformer_transform_float(dataframe: pl.DataFrame) -> None:
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out")
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


def test_column_close_transformer_transform_int() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [5, 4, 3, 2, 1],
            "col3": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
    )
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out")
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


def test_column_close_transformer_transform_atol() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, 1.00001, 1.0001, 1.001, 1.01, 1.1],
            "col2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "col3": ["a", "b", "c", "d", "e", "f"],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
    )
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out", atol=1e-3, rtol=0.0)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 1.00001, 1.0001, 1.001, 1.01, 1.1],
                "col2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "col3": ["a", "b", "c", "d", "e", "f"],
                "out": [True, True, True, False, False, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_close_transformer_transform_rtol() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, 1.00001, 1.0001, 1.001, 1.01, 1.1],
            "col2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "col3": ["a", "b", "c", "d", "e", "f"],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
    )
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out", rtol=1e-3, atol=0.0)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 1.00001, 1.0001, 1.001, 1.01, 1.1],
                "col2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "col3": ["a", "b", "c", "d", "e", "f"],
                "out": [True, True, True, False, False, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_close_transformer_transform_equal_nan_true() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, 2.0, float("nan"), 1.0, float("nan"), None, 1.0, None],
            "col2": [1.0, 1.0, 1.0, float("nan"), float("nan"), 1.0, None, None],
            "col3": ["a", "b", "c", "d", "e", "f", "g", "h"],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
    )
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out", equal_nan=True)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, float("nan"), 1.0, float("nan"), None, 1.0, None],
                "col2": [1.0, 1.0, 1.0, float("nan"), float("nan"), 1.0, None, None],
                "col3": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "out": [True, False, False, False, True, None, None, None],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_close_transformer_transform_equal_nan_false() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, 2.0, float("nan"), 1.0, float("nan"), None, 1.0, None],
            "col2": [1.0, 1.0, 1.0, float("nan"), float("nan"), 1.0, None, None],
            "col3": ["a", "b", "c", "d", "e", "f", "g", "h"],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
    )
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, float("nan"), 1.0, float("nan"), None, 1.0, None],
                "col2": [1.0, 1.0, 1.0, float("nan"), float("nan"), 1.0, None, None],
                "col3": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "out": [True, False, False, False, False, None, None, None],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_close_transformer_transform_empty() -> None:
    frame = pl.DataFrame(
        {"col1": [], "col2": []},
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnClose(actual="col1", expected="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [], "col2": [], "out": []},
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_column_close_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col1", expected="col2", out_col="col3", exist_policy="ignore")
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


def test_column_close_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col1", expected="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_close_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col1", expected="col2", out_col="col3", exist_policy="warn")
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


def test_column_close_transformer_transform_missing_policy_ignore_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col", expected="col2", out_col="out", missing_policy="ignore")
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


def test_column_close_transformer_transform_missing_policy_ignore_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(
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


def test_column_close_transformer_transform_missing_policy_raise_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col", expected="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_close_transformer_transform_missing_policy_raise_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col1", expected="missing", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_close_transformer_transform_missing_policy_warn_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(actual="col", expected="col2", out_col="out", missing_policy="warn")
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


def test_column_close_transformer_transform_missing_policy_warn_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnClose(
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
