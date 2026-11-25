import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

FEATURES = ["distance_km", "weight_kg", "hub_backlog", "pin_code_tier",
            "declared_value_rs", "transfers", "service", "destination"]
TARGET = "delivery_days"


def mae_by_group(model, df, group_col="destination"):
    """MAE of `model` on `df`, computed separately for each group value."""
    result = {}
    for group, part in df.groupby(group_col):
        result[group] = round(float(mean_absolute_error(
            part[TARGET], model.predict(part[FEATURES]))), 4)
    return result


if __name__ == "__main__":
    model = joblib.load("models/model.joblib")  # v1 (the current champion)
    later = pd.read_csv("data/raw/later.csv")
    table = mae_by_group(model, later)
    for group, value in table.items():
        print(f"{group:8s} {value}")
    print("worst served:", max(table, key=table.get))