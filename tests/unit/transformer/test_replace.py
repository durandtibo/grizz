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
    InplaceReplace,
    InplaceReplaceStrict,
    Replace,
    ReplaceStrict,
)

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
    transformer = Replace(in_col="col", out_col="out")
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'ReplaceTransformer.fit' as there are no parameters available to fit"
    )


def test_replace_transformer_equal_true() -> None:
    assert Replace(in_col="col1", out_col="out").equal(Replace(in_col="col1", out_col="out"))


def test_replace_transformer_equal_false_different_in_col() -> None:
    assert not Replace(in_col="col1", out_col="out").equal(Replace(in_col="col", out_col="out"))


def test_replace_transformer_equal_false_different_out_col() -> None:
    assert not Replace(in_col="col1", out_col="out").equal(Replace(in_col="col1", out_col="col"))


def test_replace_transformer_equal_false_different_exist_policy() -> None:
    assert not Replace(in_col="col1", out_col="out").equal(
        Replace(in_col="col1", out_col="out", exist_policy="warn")
    )


def test_replace_transformer_equal_false_different_missing_policy() -> None:
    assert not Replace(in_col="col1", out_col="out").equal(
        Replace(in_col="col1", out_col="out", missing_policy="warn")
    )


def test_replace_transformer_equal_false_different_old() -> None:
    assert not Replace(in_col="col1", out_col="out").equal(
        Replace(in_col="col1", out_col="out", old={"a": 1, "b": 2, "c": 3})
    )


def test_replace_transformer_equal_false_different_type() -> None:
    assert not Replace(in_col="col1", out_col="out").equal(42)


def test_replace_transformer_get_args() -> None:
    assert objects_are_equal(
        Replace(in_col="col1", out_col="out", old={"a": 1, "b": 2, "c": 3}).get_args(),
        {
            "in_col": "col1",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
            "old": {"a": 1, "b": 2, "c": 3},
        },
    )


def test_replace_transformer_fit_missing_policy_ignore() -> None:
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
        transformer.fit(frame)


def test_replace_transformer_fit_missing_policy_raise() -> None:
    transformer = Replace(in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.fit(frame)


def test_replace_transformer_fit_missing_policy_warn() -> None:
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
        transformer.fit(frame)


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


###############################################
#     Tests for InplaceReplaceTransformer     #
###############################################


def test_inplace_replace_transformer_repr() -> None:
    assert repr(InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3})).startswith(
        "InplaceReplaceTransformer("
    )


def test_inplace_replace_transformer_str() -> None:
    assert str(InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3})).startswith(
        "InplaceReplaceTransformer("
    )


def test_inplace_replace_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = InplaceReplace(col="col")
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'InplaceReplaceTransformer.fit' as there are no parameters available to fit"
    )


def test_inplace_replace_transformer_equal_true() -> None:
    assert InplaceReplace(col="col").equal(InplaceReplace(col="col"))


def test_inplace_replace_transformer_equal_false_different_col() -> None:
    assert not InplaceReplace(col="col").equal(InplaceReplace(col="col2"))


def test_inplace_replace_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceReplace(col="col").equal(InplaceReplace(col="col", missing_policy="warn"))


def test_inplace_replace_transformer_equal_false_different_old() -> None:
    assert not InplaceReplace(col="col").equal(
        InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3})
    )


def test_inplace_replace_transformer_equal_false_different_type() -> None:
    assert not InplaceReplace(col="col").equal(42)


def test_inplace_replace_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3}).get_args(),
        {
            "col": "col",
            "missing_policy": "raise",
            "old": {"a": 1, "b": 2, "c": 3},
        },
    )


def test_inplace_replace_transformer_fit_missing_policy_ignore() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="ignore")
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformer.fit(frame)


def test_inplace_replace_transformer_fit_missing_policy_raise() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(frame)


def test_inplace_replace_transformer_fit_missing_policy_warn() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="warn")
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(frame)


def test_inplace_replace_transformer_fit_transform() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.fit_transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": ["1", "2", "3", "d", "e"]}))


def test_inplace_replace_transformer_transform_mapping() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": ["1", "2", "3", "d", "e"]}))


def test_inplace_replace_transformer_transform_missing_policy_ignore() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="ignore")
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


def test_inplace_replace_transformer_transform_missing_policy_raise() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(frame)


def test_inplace_replace_transformer_transform_missing_policy_warn() -> None:
    transformer = InplaceReplace(col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="warn")
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
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


def test_replace_strict_transformer_equal_true() -> None:
    assert ReplaceStrict(in_col="col1", out_col="out").equal(
        ReplaceStrict(in_col="col1", out_col="out")
    )


def test_replace_strict_transformer_equal_false_different_in_col() -> None:
    assert not ReplaceStrict(in_col="col1", out_col="out").equal(
        ReplaceStrict(in_col="col", out_col="out")
    )


def test_replace_strict_transformer_equal_false_different_out_col() -> None:
    assert not ReplaceStrict(in_col="col1", out_col="out").equal(
        ReplaceStrict(in_col="col1", out_col="col")
    )


def test_replace_strict_transformer_equal_false_different_exist_policy() -> None:
    assert not ReplaceStrict(in_col="col1", out_col="out").equal(
        ReplaceStrict(in_col="col1", out_col="out", exist_policy="warn")
    )


def test_replace_strict_transformer_equal_false_different_missing_policy() -> None:
    assert not ReplaceStrict(in_col="col1", out_col="out").equal(
        ReplaceStrict(in_col="col1", out_col="out", missing_policy="warn")
    )


def test_replace_strict_transformer_equal_false_different_old() -> None:
    assert not ReplaceStrict(in_col="col1", out_col="out").equal(
        ReplaceStrict(in_col="col1", out_col="out", old={"a": 1, "b": 2, "c": 3})
    )


def test_replace_strict_transformer_equal_false_different_type() -> None:
    assert not ReplaceStrict(in_col="col1", out_col="out").equal(42)


def test_replace_strict_transformer_get_args() -> None:
    assert objects_are_equal(
        ReplaceStrict(in_col="col1", out_col="out", old={"a": 1, "b": 2, "c": 3}).get_args(),
        {
            "in_col": "col1",
            "out_col": "out",
            "exist_policy": "raise",
            "missing_policy": "raise",
            "old": {"a": 1, "b": 2, "c": 3},
        },
    )


def test_replace_strict_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = ReplaceStrict(in_col="col", out_col="out")
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'ReplaceStrictTransformer.fit' as there are no parameters available to fit"
    )


def test_replace_strict_transformer_fit_missing_policy_ignore() -> None:
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
        transformer.fit(frame)


def test_replace_strict_transformer_fit_missing_policy_raise() -> None:
    transformer = ReplaceStrict(in_col="in", out_col="out", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'in' is missing in the DataFrame"):
        transformer.fit(frame)


def test_replace_strict_transformer_fit_missing_policy_warn() -> None:
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
        transformer.fit(frame)


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


#####################################################
#     Tests for InplaceReplaceStrictTransformer     #
#####################################################


def test_inplace_replace_strict_transformer_repr() -> None:
    assert repr(InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3})).startswith(
        "InplaceReplaceStrictTransformer("
    )


def test_inplace_replace_strict_transformer_str() -> None:
    assert str(InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3})).startswith(
        "InplaceReplaceStrictTransformer("
    )


def test_inplace_replace_strict_transformer_equal_true() -> None:
    assert InplaceReplaceStrict(col="col").equal(InplaceReplaceStrict(col="col"))


def test_inplace_replace_strict_transformer_equal_false_different_col() -> None:
    assert not InplaceReplaceStrict(col="col").equal(InplaceReplaceStrict(col="col2"))


def test_inplace_replace_strict_transformer_equal_false_different_missing_policy() -> None:
    assert not InplaceReplaceStrict(col="col").equal(
        InplaceReplaceStrict(col="col", missing_policy="warn")
    )


def test_inplace_replace_strict_transformer_equal_false_different_old() -> None:
    assert not InplaceReplaceStrict(col="col").equal(
        InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3})
    )


def test_inplace_replace_strict_transformer_equal_false_different_type() -> None:
    assert not InplaceReplaceStrict(col="col").equal(42)


def test_inplace_replace_strict_transformer_get_args() -> None:
    assert objects_are_equal(
        InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3}).get_args(),
        {
            "col": "col",
            "missing_policy": "raise",
            "old": {"a": 1, "b": 2, "c": 3},
        },
    )


def test_inplace_replace_strict_transformer_fit(caplog: pytest.LogCaptureFixture) -> None:
    transformer = InplaceReplaceStrict(col="col")
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    with caplog.at_level(logging.INFO):
        transformer.fit(frame)
    assert caplog.messages[0].startswith(
        "Skipping 'InplaceReplaceStrictTransformer.fit' as there are no parameters available to fit"
    )


def test_inplace_replace_strict_transformer_fit_missing_policy_ignore() -> None:
    transformer = InplaceReplaceStrict(
        col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="ignore"
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
        transformer.fit(frame)


def test_inplace_replace_strict_transformer_fit_missing_policy_raise() -> None:
    transformer = InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.fit(frame)


def test_inplace_replace_strict_transformer_fit_missing_policy_warn() -> None:
    transformer = InplaceReplaceStrict(
        col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="warn"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
    ):
        transformer.fit(frame)


def test_inplace_replace_strict_transformer_fit_transform() -> None:
    transformer = InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.fit_transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": [1, 2, 3, 4, 5]}))


def test_inplace_replace_strict_transformer_transform_mapping() -> None:
    transformer = InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": [1, 2, 3, 4, 5]}))


def test_inplace_replace_strict_transformer_transform_mapping_default() -> None:
    transformer = InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3}, default=None)
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": [1, 2, 3, None, None]}))


def test_inplace_replace_strict_transformer_transform_missing_policy_ignore() -> None:
    transformer = InplaceReplaceStrict(
        col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="ignore"
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


def test_inplace_replace_strict_transformer_transform_missing_policy_raise() -> None:
    transformer = InplaceReplaceStrict(col="col", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.raises(ColumnNotFoundError, match="column 'col' is missing in the DataFrame"):
        transformer.transform(frame)


def test_inplace_replace_strict_transformer_transform_missing_policy_warn() -> None:
    transformer = InplaceReplaceStrict(
        col="col", old={"a": 1, "b": 2, "c": 3}, missing_policy="warn"
    )
    frame = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b"],
            "col2": ["1", "2", "3", "4", "5"],
        },
        schema={"col1": pl.String, "col2": pl.String},
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'col' is missing in the DataFrame and will be ignored"
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
