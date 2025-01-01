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
from grizz.transformer import ColumnEqual, ColumnEqualMissing


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [5, 4, 3, 2, 1],
            "col3": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
    )


############################################
#     Tests for ColumnEqualTransformer     #
############################################


def test_column_equal_transformer_repr() -> None:
    assert repr(ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_equal_transformer_str() -> None:
    assert str(ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_equal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnEqualTransformer.fit' as there are no parameters available to fit"
    )


def test_column_equal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = ColumnEqual(in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ColumnEqual(in1_col="col", in2_col="col2", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
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


def test_column_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
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


def test_column_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [True, None, None, None, False, False, True],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "out": pl.Boolean,
            },
        ),
    )


def test_column_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="col3", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": [False, False, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqual(in1_col="col1", in2_col="col2", out_col="col3", exist_policy="warn")
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col3' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": [False, False, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqual(
        in1_col="col",
        in2_col="col2",
        out_col="out",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )


def test_column_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqual(
        in1_col="col",
        in2_col="col2",
        out_col="out",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )


###################################################
#     Tests for ColumnEqualMissingTransformer     #
###################################################


def test_column_equal_missing_transformer_repr() -> None:
    assert repr(ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnEqualMissingTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_equal_missing_transformer_str() -> None:
    assert str(ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnEqualMissingTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_equal_missing_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnEqualMissingTransformer.fit' as there are no parameters available to fit"
    )


def test_column_equal_missing_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_equal_missing_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_equal_missing_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ColumnEqualMissing(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_equal_missing_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
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


def test_column_equal_missing_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
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


def test_column_equal_missing_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [True, False, False, True, False, False, True],
            },
            schema={
                "col1": pl.Float32,
                "col2": pl.Float32,
                "out": pl.Boolean,
            },
        ),
    )


def test_column_equal_missing_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(
        in1_col="col1", in2_col="col2", out_col="col3", exist_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": [False, False, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_equal_missing_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_equal_missing_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(
        in1_col="col1", in2_col="col2", out_col="col3", exist_policy="warn"
    )
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col3' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": [False, False, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_equal_missing_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(
        in1_col="col",
        in2_col="col2",
        out_col="out",
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )


def test_column_equal_missing_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_equal_missing_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnEqualMissing(
        in1_col="col",
        in2_col="col2",
        out_col="out",
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )
