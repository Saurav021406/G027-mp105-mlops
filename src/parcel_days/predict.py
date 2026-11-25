import joblib
import pandas as pd

from parcel_days.train import CATEGORICAL, NUMERIC, load_params

FEATURES = NUMERIC + CATEGORICAL


def predict_holdout(model, holdout):
    """Return a DataFrame with exactly two columns: record_id, prediction."""
    return pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": model.predict(holdout[FEATURES]),
    })


if __name__ == "__main__":
    params = load_params()
    model = joblib.load(params["paths"]["model_out"])  # the @champion model
    holdout = pd.read_csv(params["paths"]["holdout"])
    out = predict_holdout(model, holdout)
    out.to_csv("predictions/holdout_predictions.csv", index=False)
    print("rows:", len(out), "| columns:", list(out.columns))
    print(out.head(3).to_string(index=False))