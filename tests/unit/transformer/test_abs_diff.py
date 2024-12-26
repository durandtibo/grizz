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
from grizz.transformer import AbsDiffColumn


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1], "col3": ["a", "b", "c", "d", "e"]},
        schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
    )


##############################################
#     Tests for AbsDiffColumnTransformer     #
##############################################


def test_abs_diff_column_transformer_repr() -> None:
    assert str(AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")).startswith(
        "AbsDiffColumnTransformer("
    )


def test_abs_diff_column_transformer_str() -> None:
    assert str(AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")).startswith(
        "AbsDiffColumnTransformer("
    )


def test_abs_diff_column_transformer_fit(
    dataframe: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")
    with caplog.at_level(logging.INFO):
        transformer.fit(dataframe)
    assert caplog.messages[0].startswith(
        "Skipping 'AbsDiffColumnTransformer.fit' as there are no parameters available to fit"
    )


def test_abs_diff_column_transformer_fit_missing_policy_ignore_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col", in2_col="col2", out_col="diff", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_abs_diff_column_transformer_fit_missing_policy_ignore_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col1", in2_col="missing", out_col="diff", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(dataframe)


def test_abs_diff_column_transformer_fit_missing_policy_raise_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(in1_col="col", in2_col="col2", out_col="diff")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_abs_diff_column_transformer_fit_missing_policy_raise_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(in1_col="col1", in2_col="missing", out_col="diff")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        transformer.fit(dataframe)


def test_abs_diff_column_transformer_fit_missing_policy_warn_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col", in2_col="col2", out_col="diff", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignore"
    ):
        transformer.fit(dataframe)


def test_abs_diff_column_transformer_fit_missing_policy_warn_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col1", in2_col="missing", out_col="diff", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'missing' is missing in the DataFrame and will be ignore",
    ):
        transformer.fit(dataframe)


def test_abs_diff_column_transformer_fit_transform(dataframe: pl.DataFrame) -> None:
    transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")
    out = transformer.fit_transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "diff": [4, 2, 0, 2, 4],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "diff": pl.Int64},
        ),
    )


def test_abs_diff_column_transformer_transform_int(dataframe: pl.DataFrame) -> None:
    transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [5, 4, 3, 2, 1],
                "col3": ["a", "b", "c", "d", "e"],
                "diff": [4, 2, 0, 2, 4],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String, "diff": pl.Int64},
        ),
    )


def test_abs_diff_column_transformer_transform_float() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "col3": ["a", "b", "c", "d", "e"],
        },
        schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String},
    )
    transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "col3": ["a", "b", "c", "d", "e"],
                "diff": [4.0, 2.0, 0.0, 2.0, 4.0],
            },
            schema={"col1": pl.Float32, "col2": pl.Float32, "col3": pl.String, "diff": pl.Float32},
        ),
    )


def test_abs_diff_column_transformer_transform_empty() -> None:
    frame = pl.DataFrame(
        {"col1": [], "col2": []},
        schema={"col1": pl.Int64, "col2": pl.Int64},
    )
    transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="diff")
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [], "col2": [], "diff": []},
            schema={"col1": pl.Int64, "col2": pl.Int64, "diff": pl.Int64},
        ),
    )


def test_abs_diff_column_transformer_transform_exist_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
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
                "col3": [4, 2, 0, 2, 4],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64},
        ),
    )


def test_abs_diff_column_transformer_transform_exist_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="col3")
    with pytest.raises(ColumnExistsError, match="column 'col3' already exists in the DataFrame"):
        transformer.transform(dataframe)


def test_abs_diff_column_transformer_transform_exist_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(in1_col="col1", in2_col="col2", out_col="col3", exist_policy="warn")
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
                "col3": [4, 2, 0, 2, 4],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64},
        ),
    )


def test_abs_diff_column_transformer_transform_missing_policy_ignore_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col", in2_col="col2", out_col="diff", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1], "col3": ["a", "b", "c", "d", "e"]},
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )


def test_abs_diff_column_transformer_transform_missing_policy_ignore_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col1", in2_col="missing", out_col="diff", missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1], "col3": ["a", "b", "c", "d", "e"]},
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )


def test_abs_diff_column_transformer_transform_missing_policy_raise_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(in1_col="col", in2_col="col2", out_col="diff")
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_abs_diff_column_transformer_transform_missing_policy_raise_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(in1_col="col1", in2_col="missing", out_col="diff")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        transformer.transform(dataframe)


def test_abs_diff_column_transformer_transform_missing_policy_warn_in1(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col", in2_col="col2", out_col="diff", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignore"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1], "col3": ["a", "b", "c", "d", "e"]},
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )


def test_abs_diff_column_transformer_transform_missing_policy_warn_in2(
    dataframe: pl.DataFrame,
) -> None:
    transformer = AbsDiffColumn(
        in1_col="col1", in2_col="missing", out_col="diff", missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'missing' is missing in the DataFrame and will be ignore",
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1], "col3": ["a", "b", "c", "d", "e"]},
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.String},
        ),
    )
