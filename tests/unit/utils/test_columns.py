from grizz.utils.column import find_missing_columns

##########################################
#     Tests for find_missing_columns     #
##########################################


def test_find_missing_columns_1() -> None:
    assert find_missing_columns(["col1", "col2", "col3"], ["col1"]) == []


def test_find_missing_columns_2() -> None:
    assert find_missing_columns(["col1", "col2", "col3"], ["col1", "col2"]) == []


def test_find_missing_columns_3() -> None:
    assert find_missing_columns(["col1", "col2", "col3"], ["col1", "col2", "col3"]) == []


def test_find_missing_columns_4() -> None:
    assert find_missing_columns(["col1", "col2", "col3"], ["col1", "col2", "col3", "col4"]) == [
        "col4"
    ]


def test_find_missing_columns_empty() -> None:
    assert find_missing_columns([], []) == []
