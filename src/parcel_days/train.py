import hashlib
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from parcel_days.clean import clean_data

NUMERIC = ["distance_km", "weight_kg", "hub_backlog",
           "pin_code_tier", "declared_value_rs", "transfers"]
CATEGORICAL = ["service", "destination"]
TARGET = "delivery_days"


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(n_estimators, max_depth, random_state):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    return Pipeline([("pre", pre), ("model", model)])


def main():
    params = load_params()
    train_path = params["paths"]["train"]
    seed = params["train"]["random_state"]

    df = clean_data(pd.read_csv(train_path))
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=params["train"]["test_size"], random_state=seed
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    data_hash = sha256_of(train_path)

    best = None
    for i, cfg in enumerate(params["train"]["runs"]):
        with mlflow.start_run(run_name=f"rf_{i}") as run:
            mlflow.log_params(cfg)
            mlflow.log_param("data_sha256", data_hash)
            pipe = build_pipeline(cfg["n_estimators"], cfg["max_depth"], seed)
            pipe.fit(X_tr, y_tr)
            mae = mean_absolute_error(y_va, pipe.predict(X_va))
            mlflow.log_metric("mae_valid", mae)
            info = mlflow.sklearn.log_model(
                pipe, name="model", serialization_format="cloudpickle"
            )
            print(f"run {run.info.run_id}  cfg={cfg}  mae_valid={mae:.4f}")
            if best is None or mae < best["mae"]:
                best = {"mae": mae, "run_id": run.info.run_id,
                        "uri": info.model_uri, "pipe": pipe}

    name = params["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(best["uri"], name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)

    out = params["paths"]["model_out"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(best["pipe"], out)
    print("BEST run id:", best["run_id"], " mae:", round(best["mae"], 4))
    print("registered", name, "version", mv.version, "-> alias champion")


if __name__ == "__main__":
    main()