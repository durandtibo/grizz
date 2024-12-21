from __future__ import annotations

import pytest

from grizz.transformer.columns import check_missing_columns

###########################################
#     Tests for check_missing_columns     #
###########################################


@pytest.mark.parametrize("ignore_missing", [True, False])
def test_check_missing_columns_no_missing(ignore_missing: bool) -> None:
    check_missing_columns(missing_cols=[], ignore_missing=ignore_missing)


def test_check_missing_columns_missing_ignore_missing_true() -> None:
    with pytest.warns(
        RuntimeWarning, match="1 columns are missing in the DataFrame and will be ignored:"
    ):
        check_missing_columns(missing_cols=["col5"], ignore_missing=True)


def test_check_missing_columns_missing_ignore_missing_false() -> None:
    with pytest.raises(RuntimeError, match="1 columns are missing in the DataFrame:"):
        check_missing_columns(missing_cols=["col5"])
