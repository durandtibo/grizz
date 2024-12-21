from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from grizz.transformer import FillNan, FillNull


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, None],
            "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
            "col3": ["a", "b", "c", "d", None],
            "col4": [1.2, float("nan"), 3.2, None, 5.2],
        },
        schema={
            "col1": pl.Int64,
            "col2": pl.Float64,
            "col3": pl.String,
            "col4": pl.Float64,
        },
    )


########################################
#     Tests for FillNanTransformer     #
########################################


def test_fill_nan_transformer_repr() -> None:
    assert repr(FillNan(columns=["col1", "col4"])) == (
        "FillNanTransformer(columns=('col1', 'col4'), ignore_missing=False)"
    )


def test_fill_nan_transformer_repr_with_kwargs() -> None:
    assert repr(FillNan(columns=["col1", "col4"], value=100)) == (
        "FillNanTransformer(columns=('col1', 'col4'), ignore_missing=False, value=100)"
    )


def test_fill_nan_transformer_str() -> None:
    assert str(FillNan(columns=["col1", "col4"])) == (
        "FillNanTransformer(columns=('col1', 'col4'), ignore_missing=False)"
    )


def test_fill_nan_transformer_str_with_kwargs() -> None:
    assert str(FillNan(columns=["col1", "col4"], value=100)) == (
        "FillNanTransformer(columns=('col1', 'col4'), ignore_missing=False, value=100)"
    )


def test_fill_nan_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col1", "col4"], value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, 100.0],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_transform_empty() -> None:
    transformer = FillNan(columns=[], value=100)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_fill_nan_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.String})
    transformer = FillNan(columns=["col"], value=100)
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": []}, schema={"col": pl.String}))


def test_fill_nan_transformer_transform_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], value=100)
    with pytest.raises(RuntimeError, match="1 columns are missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_fill_nan_transformer_transform_ignore_missing_true(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col1", "col4", "col5"], ignore_missing=True, value=100)
    with pytest.warns(
        RuntimeWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, None],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, 100.0, 3.2, None, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_fill_nan_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_fill_nan_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(value=100)
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_nan_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_fill_nan_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(value=100)
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_nan_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_fill_nan_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNan(value=100)
    assert transformer.find_missing_columns(dataframe) == ()


#########################################
#     Tests for FillNullTransformer     #
#########################################


def test_fill_null_transformer_repr() -> None:
    assert repr(FillNull(columns=["col1", "col4"])) == (
        "FillNullTransformer(columns=('col1', 'col4'), ignore_missing=False)"
    )


def test_fill_null_transformer_repr_with_kwargs() -> None:
    assert repr(FillNull(columns=["col1", "col4"], value=100)) == (
        "FillNullTransformer(columns=('col1', 'col4'), ignore_missing=False, value=100)"
    )


def test_fill_null_transformer_str() -> None:
    assert str(FillNull(columns=["col1", "col4"])) == (
        "FillNullTransformer(columns=('col1', 'col4'), ignore_missing=False)"
    )


def test_fill_null_transformer_str_with_kwargs() -> None:
    assert str(FillNull(columns=["col1", "col4"], value=100)) == (
        "FillNullTransformer(columns=('col1', 'col4'), ignore_missing=False, value=100)"
    )


def test_fill_null_transformer_transform(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col1", "col4"], value=100)
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 100],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_transform_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(value=100)
    out = transformer.transform(dataframe.drop("col3"))
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 100],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col4": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col4": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_transform_empty() -> None:
    transformer = FillNull(columns=[], value=100)
    out = transformer.transform(pl.DataFrame({}))
    assert_frame_equal(out, pl.DataFrame({}))


def test_fill_null_transformer_transform_empty_row() -> None:
    frame = pl.DataFrame({"col": []}, schema={"col": pl.String})
    transformer = FillNull(columns=["col"], value=100)
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": []}, schema={"col": pl.String}))


def test_fill_null_transformer_transform_ignore_missing_false(
    dataframe: pl.DataFrame,
) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], value=100)
    with pytest.raises(RuntimeError, match="1 columns are missing in the DataFrame:"):
        transformer.transform(dataframe)


def test_fill_null_transformer_transform_ignore_missing_true(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col1", "col4", "col5"], ignore_missing=True, value=100)
    with pytest.warns(
        RuntimeWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 100],
                "col2": [1.2, 2.2, 3.2, 4.2, float("nan")],
                "col3": ["a", "b", "c", "d", None],
                "col4": [1.2, float("nan"), 3.2, 100.0, 5.2],
            },
            schema={
                "col1": pl.Int64,
                "col2": pl.Float64,
                "col3": pl.String,
                "col4": pl.Float64,
            },
        ),
    )


def test_fill_null_transformer_find_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_columns(dataframe) == ("col2", "col3", "col5")


def test_fill_null_transformer_find_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(value=100)
    assert transformer.find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_null_transformer_find_common_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_common_columns(dataframe) == ("col2", "col3")


def test_fill_null_transformer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(value=100)
    assert transformer.find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_fill_null_transformer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(columns=["col2", "col3", "col5"], value=100)
    assert transformer.find_missing_columns(dataframe) == ("col5",)


def test_fill_null_transformer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    transformer = FillNull(value=100)
    assert transformer.find_missing_columns(dataframe) == ()
