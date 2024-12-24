from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from iden.io import save_text

from grizz.exceptions import DataFrameNotFoundError
from grizz.ingestor.utils import check_dataframe_file

if TYPE_CHECKING:
    from pathlib import Path

##########################################
#     Tests for check_dataframe_file     #
##########################################


def test_check_dataframe_file_exists(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("meow", path)
    check_dataframe_file(path)


def test_check_dataframe_file_missing(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    with pytest.raises(DataFrameNotFoundError, match="DataFrame file does not exist"):
        check_dataframe_file(path)
