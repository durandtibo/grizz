from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from iden.io import save_text

from grizz.exceptions import DataNotFoundError
from grizz.ingestor.utils import check_data_file

if TYPE_CHECKING:
    from pathlib import Path

#####################################
#     Tests for check_data_file     #
#####################################


def test_check_data_file_exists(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("meow", path)
    check_data_file(path)


def test_check_data_file_missing(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    with pytest.raises(DataNotFoundError, match=r"Data file does not exist"):
        check_data_file(path)
