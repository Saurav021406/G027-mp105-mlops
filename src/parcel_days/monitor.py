import numpy as np
import pandas as pd

from parcel_days.clean import clean_data

NUMERIC = ["distance_km", "weight_kg", "hub_backlog",
           "pin_code_tier", "declared_value_rs", "transfers"]


def psi(expected, actual, bins=10):
    """Population Stability Index between two samples.

    Bin edges come from the expected (train) sample: `bins` equal-width bins
    over its min..max. The outer edges are opened to -inf/+inf so that values
    outside the train range are still counted.
    """
    edges = np.histogram_bin_edges(expected, bins=bins)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-4  # avoids log(0) and division by zero for empty bins
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(train, later):
    return {c: round(psi(train[c], later[c]), 4) for c in NUMERIC}


if __name__ == "__main__":
    train = clean_data(pd.read_csv("data/raw/train.csv"))
    later = pd.read_csv("data/raw/later.csv")
    table = psi_table(train, later)
    for col, value in table.items():
        print(f"{col:20s} {value}")
    print("most drifted:", max(table, key=table.get))