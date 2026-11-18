import pandas as pd

from parcel_days.clean import clean_data
from parcel_days.data_checks import find_target_typos, find_unit_mixup


def make_df():
    return pd.DataFrame({
        "record_id": ["a", "b", "c", "c"],
        "date": ["2026-01-01"] * 4,
        "distance_km": [100, 200, 300, 300],
        "weight_kg": [5000.0, 10.0, 12.0, 12.0],
        "hub_backlog": [10, 20, 30, 30],
        "pin_code_tier": [1, 2, 3, 3],
        "declared_value_rs": [1000, 2000, 3000, 3000],
        "transfers": [1, 2, 3, 3],
        "service": [" Express", "exp.", "standard", "standard"],
        "destination": ["Metro", "Rural", "Tier-2", "Tier-2"],
        "delivery_days": [3.0, 40.0, 5.0, 5.0],
    })


def test_clean_removes_duplicates():
    assert len(clean_data(make_df())) == 3


def test_clean_fixes_units_and_typos():
    out = clean_data(make_df())
    assert out["weight_kg"].max() < 100
    assert out["delivery_days"].max() < 10


def test_clean_fixes_service_spelling():
    out = clean_data(make_df())
    assert set(out["service"]) <= {"express", "standard", "economy"}


def test_find_target_typos_counts():
    assert find_target_typos(make_df()) == 1