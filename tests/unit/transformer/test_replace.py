from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from grizz.transformer import Replace, ReplaceStrict

########################################
#     Tests for ReplaceTransformer     #
########################################


def test_replace_dataframe_transformer_repr() -> None:
    assert repr(
        Replace(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    ).startswith("ReplaceTransformer(")


def test_replace_dataframe_transformer_str() -> None:
    assert str(
        Replace(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    ).startswith("ReplaceTransformer(")


def test_replace_dataframe_transformer_transform_mapping() -> None:
    transformer = Replace(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": ["1", "2", "3", "d", "e"]})
    )


def test_replace_dataframe_transformer_transform_same_column() -> None:
    transformer = Replace(orig_column="col", final_column="col", old={"a": 1, "b": 2, "c": 3})
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": ["1", "2", "3", "d", "e"]}))


#######################################################
#     Tests for ReplaceStrictTransformer     #
#######################################################


def test_replace_strict_dataframe_transformer_repr() -> None:
    assert repr(
        ReplaceStrict(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    ).startswith("ReplaceStrictTransformer(")


def test_replace_strict_dataframe_transformer_str() -> None:
    assert str(
        ReplaceStrict(orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3})
    ).startswith("ReplaceStrictTransformer(")


def test_replace_strict_dataframe_transformer_transform_mapping() -> None:
    transformer = ReplaceStrict(
        orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": [1, 2, 3, 4, 5]})
    )


def test_replace_strict_dataframe_transformer_transform_mapping_default() -> None:
    transformer = ReplaceStrict(
        orig_column="old", final_column="new", old={"a": 1, "b": 2, "c": 3}, default=None
    )
    frame = pl.DataFrame({"old": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pl.DataFrame({"old": ["a", "b", "c", "d", "e"], "new": [1, 2, 3, None, None]})
    )


def test_replace_strict_dataframe_transformer_transform_same_column() -> None:
    transformer = ReplaceStrict(
        orig_column="col", final_column="col", old={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )
    frame = pl.DataFrame({"col": ["a", "b", "c", "d", "e"]})
    out = transformer.transform(frame)
    assert_frame_equal(out, pl.DataFrame({"col": [1, 2, 3, 4, 5]}))
