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
from grizz.transformer import (
    ColumnEqual,
    ColumnEqualMissing,
    ColumnGreater,
    ColumnGreaterEqual,
    ColumnLower,
    ColumnLowerEqual,
    ColumnNotEqual,
    ColumnNotEqualMissing,
)


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


def test_column_equal_transformer_equal_true() -> None:
    assert ColumnEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqual(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_equal_transformer_equal_false_different_in1_col() -> None:
    assert not ColumnEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqual(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_equal_transformer_equal_false_different_in2_col() -> None:
    assert not ColumnEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqual(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_equal_transformer_equal_false_different_out_col() -> None:
    assert not ColumnEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqual(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_equal_transformer_equal_false_different_exist_policy() -> None:
    assert not ColumnEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqual(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_equal_transformer_equal_false_different_missing_policy() -> None:
    assert not ColumnEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqual(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_equal_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnEqual(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
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
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
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


def test_column_equal_missing_transformer_equal_true() -> None:
    assert ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_equal_missing_transformer_equal_false_different_in1_col() -> None:
    assert not ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqualMissing(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_equal_missing_transformer_equal_false_different_in2_col() -> None:
    assert not ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqualMissing(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_equal_missing_transformer_equal_false_different_out_col() -> None:
    assert not ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_equal_missing_transformer_equal_false_different_exist_policy() -> None:
    assert not ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_equal_missing_transformer_equal_false_different_missing_policy() -> None:
    assert not ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_equal_missing_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnEqualMissing(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_equal_missing_missing_transformer_fit(
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
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
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


###################################################
#     Tests for ColumnGreaterEqualTransformer     #
###################################################


def test_column_greater_equal_transformer_repr() -> None:
    assert repr(ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnGreaterEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_greater_equal_transformer_str() -> None:
    assert str(ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnGreaterEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_greater_equal_transformer_equal_true() -> None:
    assert ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_greater_equal_transformer_equal_false_different_in1_col() -> None:
    assert not ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreaterEqual(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_greater_equal_transformer_equal_false_different_in2_col() -> None:
    assert not ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreaterEqual(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_greater_equal_transformer_equal_false_different_out_col() -> None:
    assert not ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_greater_equal_transformer_equal_false_different_exist_policy() -> None:
    assert not ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_greater_equal_transformer_equal_false_different_missing_policy() -> None:
    assert not ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_greater_equal_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_greater_equal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnGreaterEqualTransformer.fit' as there are no parameters available to fit"
    )


def test_column_greater_equal_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_greater_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_greater_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ColumnGreaterEqual(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_greater_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [False, False, True, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_greater_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [False, False, True, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_greater_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [True, None, None, None, True, False, True],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_column_greater_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(
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
                "col3": [False, False, True, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_greater_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_greater_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(
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
                "col3": [False, False, True, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_greater_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(
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


def test_column_greater_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_greater_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreaterEqual(
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


##############################################
#     Tests for ColumnGreaterTransformer     #
##############################################


def test_column_greater_transformer_repr() -> None:
    assert repr(ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnGreaterTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_greater_transformer_str() -> None:
    assert str(ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnGreaterTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_greater_equal_true() -> None:
    assert ColumnGreater(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_greater_equal_false_different_in1_col() -> None:
    assert not ColumnGreater(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreater(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_greater_equal_false_different_in2_col() -> None:
    assert not ColumnGreater(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreater(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_greater_equal_false_different_out_col() -> None:
    assert not ColumnGreater(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreater(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_greater_equal_false_different_exist_policy() -> None:
    assert not ColumnGreater(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreater(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_greater_equal_false_different_missing_policy() -> None:
    assert not ColumnGreater(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnGreater(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_greater_get_args() -> None:
    assert objects_are_equal(
        ColumnGreater(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_greater_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnGreaterTransformer.fit' as there are no parameters available to fit"
    )


def test_column_greater_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_greater_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_greater_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ColumnGreater(in1_col="col", in2_col="col2", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_greater_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [False, False, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_greater_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [False, False, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_greater_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnGreater(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [False, None, None, None, True, False, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_column_greater_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(
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
                "col3": [False, False, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_greater_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_greater_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(in1_col="col1", in2_col="col2", out_col="col3", exist_policy="warn")
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
                "col3": [False, False, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_greater_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(
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


def test_column_greater_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_greater_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnGreater(
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


#################################################
#     Tests for ColumnLowerEqualTransformer     #
#################################################


def test_column_lower_equal_transformer_repr() -> None:
    assert repr(ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnLowerEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_lower_equal_transformer_str() -> None:
    assert str(ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnLowerEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_lower_equal_transformer_equal_true() -> None:
    assert ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_lower_equal_transformer_equal_false_different_in1_col() -> None:
    assert not ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLowerEqual(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_lower_equal_transformer_equal_false_different_in2_col() -> None:
    assert not ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLowerEqual(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_lower_equal_transformer_equal_false_different_out_col() -> None:
    assert not ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_lower_equal_transformer_equal_false_different_exist_policy() -> None:
    assert not ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_lower_equal_transformer_equal_false_different_missing_policy() -> None:
    assert not ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_lower_equal_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_lower_equal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnLowerEqualTransformer.fit' as there are no parameters available to fit"
    )


def test_column_lower_equal_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_lower_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_lower_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ColumnLowerEqual(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_lower_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_lower_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_lower_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [True, None, None, None, False, True, True],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_column_lower_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(
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
                "col3": [True, True, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_lower_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_lower_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(
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
                "col3": [True, True, True, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_lower_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(
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


def test_column_lower_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_lower_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLowerEqual(
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


############################################
#     Tests for ColumnLowerTransformer     #
############################################


def test_column_lower_transformer_repr() -> None:
    assert repr(ColumnLower(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnLowerTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_lower_transformer_str() -> None:
    assert str(ColumnLower(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnLowerTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_lower_transformer_equal_true() -> None:
    assert ColumnLower(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLower(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_lower_transformer_equal_false_different_in1_col() -> None:
    assert not ColumnLower(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLower(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_lower_transformer_equal_false_different_in2_col() -> None:
    assert not ColumnLower(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLower(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_lower_transformer_equal_false_different_out_col() -> None:
    assert not ColumnLower(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLower(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_lower_transformer_equal_false_different_exist_policy() -> None:
    assert not ColumnLower(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLower(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_lower_transformer_equal_false_different_missing_policy() -> None:
    assert not ColumnLower(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnLower(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_lower_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnLower(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_lower_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnLowerTransformer.fit' as there are no parameters available to fit"
    )


def test_column_lower_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_lower_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_lower_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ColumnLower(in1_col="col", in2_col="col2", out_col="out", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_lower_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, False, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_lower_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, False, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_lower_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [False, None, None, None, False, True, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_column_lower_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="col3", exist_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": [True, True, False, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_lower_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_lower_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(in1_col="col1", in2_col="col2", out_col="col3", exist_policy="warn")
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
                "col3": [True, True, False, False, False],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_lower_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(
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


def test_column_lower_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_lower_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnLower(
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


###############################################
#     Tests for ColumnNotEqualTransformer     #
###############################################


def test_column_not_equal_transformer_repr() -> None:
    assert repr(ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnNotEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_not_equal_transformer_str() -> None:
    assert str(ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnNotEqualTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_not_equal_transformer_equal_true() -> None:
    assert ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_not_equal_transformer_equal_false_different_in1_col() -> None:
    assert not ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqual(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_not_equal_transformer_equal_false_different_in2_col() -> None:
    assert not ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqual(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_not_equal_transformer_equal_false_different_out_col() -> None:
    assert not ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_not_equal_transformer_equal_false_different_exist_policy() -> None:
    assert not ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_not_equal_transformer_equal_false_different_missing_policy() -> None:
    assert not ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_not_equal_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_not_equal_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnNotEqualTransformer.fit' as there are no parameters available to fit"
    )


def test_column_not_equal_transformer_fit_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    transformer = ColumnNotEqual(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_not_equal_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_not_equal_transformer_fit_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    transformer = ColumnNotEqual(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_not_equal_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_not_equal_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_not_equal_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [False, None, None, None, True, True, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_column_not_equal_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqual(
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
                "col3": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_not_equal_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqual(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_not_equal_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqual(
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
                "col3": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_not_equal_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqual(
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


def test_column_not_equal_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqual(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_not_equal_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqual(
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


######################################################
#     Tests for ColumnNotEqualMissingTransformer     #
######################################################


def test_column_not_equal_missing_transformer_repr() -> None:
    assert repr(ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnNotEqualMissingTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_not_equal_missing_transformer_str() -> None:
    assert str(ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")) == (
        "ColumnNotEqualMissingTransformer(in1_col='col1', in2_col='col2', out_col='out', "
        "exist_policy='raise', missing_policy='raise')"
    )


def test_column_not_equal_missing_transformer_equal_true() -> None:
    assert ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    )


def test_column_not_equal_missing_transformer_equal_false_different_in1_col() -> None:
    assert not ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqualMissing(in1_col="col", in2_col="col2", out_col="out")
    )


def test_column_not_equal_missing_transformer_equal_false_different_in2_col() -> None:
    assert not ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqualMissing(in1_col="col1", in2_col="col", out_col="out")
    )


def test_column_not_equal_missing_transformer_equal_false_different_out_col() -> None:
    assert not ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="col")
    )


def test_column_not_equal_missing_transformer_equal_false_different_exist_policy() -> None:
    assert not ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out", exist_policy="warn")
    )


def test_column_not_equal_missing_transformer_equal_false_different_missing_policy() -> None:
    assert not ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out").equal(
        ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out", missing_policy="warn")
    )


def test_column_not_equal_missing_transformer_get_args() -> None:
    assert objects_are_equal(
        ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out").get_args(),
        {
            "in1_col": "col1",
            "in2_col": "col2",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
        },
    )


def test_column_not_equal_missing_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'ColumnNotEqualMissingTransformer.fit' as there are no parameters available to fit"
    )


def test_column_not_equal_missing_transformer_fit_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_column_not_equal_missing_transformer_fit_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_column_not_equal_missing_transformer_fit_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(
        in1_col="col", in2_col="col2", out_col="out", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(dataframe)


def test_column_not_equal_missing_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_not_equal_missing_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "out": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "out": pl.Boolean},
        ),
    )


def test_column_not_equal_missing_transformer_transform_nulls_and_nans() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
            "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32},
    )
    transformer = ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="out")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, None, 1.0, None, float("nan"), -1.0, float("nan")],
                "col2": [1.0, 1.0, None, None, -1.0, float("nan"), float("nan")],
                "out": [False, True, True, False, True, True, False],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "out": pl.Boolean},
        ),
    )


def test_column_not_equal_missing_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(
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
                "col3": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_not_equal_missing_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_column_not_equal_missing_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(
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
                "col3": [True, True, False, True, True],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Boolean},
        ),
    )


def test_column_not_equal_missing_transformer_transform_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(
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


def test_column_not_equal_missing_transformer_transform_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(in1_col="col", in2_col="col2", out_col="out")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_column_not_equal_missing_transformer_transform_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = ColumnNotEqualMissing(
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
