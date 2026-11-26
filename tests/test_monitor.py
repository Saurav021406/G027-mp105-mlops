import numpy as np
import pandas as pd

from parcel_days.monitor import psi
from parcel_days.retrain import split_later


def test_psi_same_data_is_about_zero():
    x = np.random.default_rng(0).normal(100, 10, 1000)
    assert psi(x, x) < 0.01


def test_psi_big_shift_is_large():
    x = np.random.default_rng(0).normal(100, 10, 1000)
    assert psi(x, x + 50) > 1


def test_split_later_sizes_and_sorted_by_date():
    dates = pd.date_range("2026-09-01", periods=10).strftime("%Y-%m-%d")
    later = pd.DataFrame({"date": dates[::-1], "value": range(10)})

    train_part, check_part = split_later(later, 0.7)

    assert len(train_part) == 7
    assert len(check_part) == 3
    combined = pd.concat([train_part, check_part])
    assert list(combined["date"]) == sorted(dates)
    assert train_part["date"].max() < check_part["date"].min()
