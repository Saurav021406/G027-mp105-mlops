import pandas as pd

from parcel_days.data_checks import (
    find_duplicates,
    find_leaking_column,
    find_sentinel,
    find_spelling_variants,
    find_stuck_sensor,
)


def test_find_duplicates_counts_extra_copies():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert find_duplicates(df) == 1


def test_find_spelling_variants_maps_to_three_levels():
    df = pd.DataFrame({"service": ["express", "exp.", " Express", "STANDARD", "sta.", "eco.", "economy"]})
    mapping = find_spelling_variants(df)
    assert set(mapping.values()) == {"express", "standard", "economy"}


def test_find_stuck_sensor_finds_longest_run():
    dates = pd.date_range("2026-05-01", periods=8).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "declared_value_rs": [5, 7, 9, 9, 9, 9, 3, 4]})
    col, value, start, end, length = find_stuck_sensor(df)
    assert (col, value, start, end, length) == ("declared_value_rs", 9, "2026-05-03", "2026-05-06", 4)


def test_find_sentinel_detects_and_ignores():
    df = pd.DataFrame({"x": [1, -999, 3], "y": [1, 2, 3]})
    assert find_sentinel(df) == {"x": {"value": -999, "count": 1}}


def test_find_leaking_column_flags_target_copy():
    train = pd.DataFrame({"f": [1, 2, 3, 4], "leak": [2, 4, 6, 8], "delivery_days": [1, 2, 3, 4]})
    holdout = pd.DataFrame({"f": [1, 2]})
    assert "leak" in find_leaking_column(train, holdout)
    assert find_leaking_column(train, train.drop(columns=["delivery_days"])) == {}