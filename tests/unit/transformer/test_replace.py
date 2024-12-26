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
from grizz.transformer import Replace, ReplaceStrict

########################################
#     Tests for ReplaceTransformer     #
########################################


def test_replace_transformer_repr() -> None:
    assert repr(Replace(in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3})).startswith(
        "ReplaceTransformer("
    )


def test_replace_transformer_str() -> None:
    assert str(Replace(in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3})).startswith(
        "ReplaceTransformer("
    )


def test_replace_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = Replace(in_col="col1", out_col="out")
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'ReplaceTransformer.fit' as there are no parameters available to fit"
    )


def test_replace_transformer_fit_transform() -> None:
    transformer = Replace(in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.fit_transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": ["1", "2", "3", "d", "e"]})
    )


def test_replace_transformer_transform_mapping() -> None:
    transformer = Replace(in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": ["1", "2", "3", "d", "e"]})
    )


def test_replace_transformer_transform_same_column() -> None:
    transformer = Replace(
        in_col="col", out_col="col", old={"a": 1, "b": 2, "c": 3}, exist_policy="ignore"
    )
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": ["1", "2", "3", "d", "e"]}))


def test_replace_transformer_transform_missing_policy_ignore() -> None:
    transformer = Replace(
        in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3}, missing_policy="ignore"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": ["1", "2", "3", "4", "5"],
            },
            schema={"col1": pl.String, "col2": pl.String},
        ),
    )


def test_replace_transformer_transform_missing_policy_raise() -> None:
    transformer = Replace(in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.transform(frame)


def test_replace_transformer_transform_missing_policy_warn() -> None:
    transformer = Replace(
        in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3}, missing_policy="warn"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": ["1", "2", "3", "4", "5"],
            },
            schema={"col1": pl.String, "col2": pl.String},
        ),
    )


def test_replace_transformer_transform_exist_policy_ignore() -> None:
    transformer = Replace(
        in_col="col1", out_col="col2", old={"a": 1, "b": 2, "c": 3}, exist_policy="ignore"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": ["1", "2", "3", "1", "2"],
            },
            schema={"col1": pl.String, "col2": pl.String},
        ),
    )


def test_replace_transformer_transform_exist_policy_raise() -> None:
    transformer = Replace(in_col="col1", out_col="col2", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(frame)


def test_replace_transformer_transform_exist_policy_warn() -> None:
    transformer = Replace(
        in_col="col1", out_col="col2", old={"a": 1, "b": 2, "c": 3}, exist_policy="warn"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": ["1", "2", "3", "1", "2"],
            },
            schema={"col1": pl.String, "col2": pl.String},
        ),
    )


##############################################
#     Tests for ReplaceStrictTransformer     #
##############################################


def test_replace_strict_transformer_repr() -> None:
    assert repr(
        ReplaceStrict(in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3})
    ).startswith("ReplaceStrictTransformer(")


def test_replace_strict_transformer_str() -> None:
    assert str(ReplaceStrict(in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3})).startswith(
        "ReplaceStrictTransformer("
    )


def test_replace_strict_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = ReplaceStrict(in_col="col1", out_col="out")
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'ReplaceStrictTransformer.fit' as there are no parameters available to fit"
    )


def test_replace_strict_transformer_fit_transform() -> None:
    transformer = ReplaceStrict(
        in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.fit_transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": [1, 2, 3, 4, 5]})
    )


def test_replace_strict_transformer_transform_mapping() -> None:
    transformer = ReplaceStrict(
        in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": [1, 2, 3, 4, 5]})
    )


def test_replace_strict_transformer_transform_mapping_default() -> None:
    transformer = ReplaceStrict(
        in_col="old", out_col="new", old={"a": 1, "b": 2, "c": 3}, default=None
    )
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": [1, 2, 3, None, None]})
    )


def test_replace_strict_transformer_transform_same_column() -> None:
    transformer = ReplaceStrict(
        in_col="col",
        out_col="col",
        old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        exist_policy="ignore",
    )
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": [1, 2, 3, 4, 5]}))


def test_replace_strict_transformer_transform_missing_policy_ignore() -> None:
    transformer = ReplaceStrict(
        in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3}, missing_policy="ignore"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": ["1", "2", "3", "4", "5"],
            },
            schema={"col1": pl.String, "col2": pl.String},
        ),
    )


def test_replace_strict_transformer_transform_missing_policy_raise() -> None:
    transformer = ReplaceStrict(in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.transform(frame)


def test_replace_strict_transformer_transform_missing_policy_warn() -> None:
    transformer = ReplaceStrict(
        in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3}, missing_policy="warn"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'in' is missing in the DataFrame and will be ignored"
    ):
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": ["1", "2", "3", "4", "5"],
            },
            schema={"col1": pl.String, "col2": pl.String},
        ),
    )


def test_replace_strict_transformer_transform_exist_policy_ignore() -> None:
    transformer = ReplaceStrict(
        in_col="col1", out_col="col2", old={"a": 1, "b": 2, "c": 3}, exist_policy="ignore"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": [1, 2, 3, 1, 2],
            },
            schema={"col1": pl.String, "col2": pl.Int64},
        ),
    )


def test_replace_strict_transformer_transform_exist_policy_raise() -> None:
    transformer = ReplaceStrict(in_col="col1", out_col="col2", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnExistsError, match="column 'col2' already exists in the DataFrame"):
        transformer.transform(frame)


def test_replace_strict_transformer_transform_exist_policy_warn() -> None:
    transformer = ReplaceStrict(
        in_col="col1", out_col="col2", old={"a": 1, "b": 2, "c": 3}, exist_policy="warn"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnExistsWarning,
        match="column 'col2' already exists in the DataFrame and will be overwritten",
    ):
        out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": ["a", "b", "c", "a", "b"],
                "col2": [1, 2, 3, 1, 2],
            },
            schema={"col1": pl.String, "col2": pl.Int64},
        ),
    )
