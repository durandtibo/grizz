from __future__ import annotations

from grizz.transformer import FillNan, FillNanTransformer
from grizz.transformer.utils import get_classname, message_skip_fit

###################################
#     Tests for get_classname     #
###################################


def test_get_classname() -> None:
    assert (
        get_classname(FillNanTransformer(columns=["col1", "col4"], prefix="", suffix="_out"))
        == "FillNanTransformer"
    )


def test_get_classname_short() -> None:
    assert (
        get_classname(FillNan(columns=["col1", "col4"], prefix="", suffix="_out"))
        == "FillNanTransformer"
    )


######################################
#     Tests for message_skip_fit     #
######################################


def test_message_skip_fit() -> None:
    assert (
        message_skip_fit("FillNanTransformer")
        == "Skipping 'FillNanTransformer.fit' as there are no parameters available to fit"
    )
