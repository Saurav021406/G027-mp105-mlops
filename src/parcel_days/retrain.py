import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from parcel_days.clean import clean_data
from parcel_days.train import (CATEGORICAL, NUMERIC, TARGET, build_pipeline,
                               load_params, sha256_of)

FEATURES = NUMERIC + CATEGORICAL


def split_later(later, train_fraction):
    """Sort later.csv by date; first part is for training v2, last part is the check set."""
    later = later.sort_values("date", kind="stable").reset_index(drop=True)
    cut = int(len(later) * train_fraction)
    return later.iloc[:cut], later.iloc[cut:]


def main():
    params = load_params()
    rt = params["retrain"]
    name = params["mlflow"]["registered_model_name"]

    train = clean_data(pd.read_csv(params["paths"]["train"]))
    later = pd.read_csv(params["paths"]["later"])
    later_train, check = split_later(later, rt["train_fraction"])

    v1 = joblib.load("models/model_v1.joblib")
    mae_v1 = mean_absolute_error(check[TARGET], v1.predict(check[FEATURES]))

    combined = pd.concat([train, later_train], ignore_index=True)
    v2 = build_pipeline(rt["n_estimators"], rt["max_depth"], params["train"]["random_state"])
    v2.fit(combined[FEATURES], combined[TARGET])
    mae_v2 = mean_absolute_error(check[TARGET], v2.predict(check[FEATURES]))

    improvement = (mae_v1 - mae_v2) / mae_v1
    promote = improvement >= rt["promote_if_better_by"]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_params({"n_estimators": rt["n_estimators"], "max_depth": rt["max_depth"],
                           "rows_train": len(combined), "rows_check": len(check)})
        mlflow.log_param("data_sha256", sha256_of(params["paths"]["train"]))
        mlflow.log_metric("mae_v1_check", mae_v1)
        mlflow.log_metric("mae_v2_check", mae_v2)
        mlflow.log_metric("mae_improvement", improvement)
        info = mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
    version = mlflow.register_model(info.model_uri, name).version

    if promote:
        MlflowClient().set_registered_model_alias(name, "champion", version)
        joblib.dump(v2, params["paths"]["model_out"])

    print(f"v1 MAE on check: {mae_v1:.4f}")
    print(f"v2 MAE on check: {mae_v2:.4f}")
    print(f"improvement: {improvement:.1%} (need {rt['promote_if_better_by']:.0%})")
    print("decision:", "promote" if promote else "keep")
    print("v2 registered as version", version, "| run id:", run.info.run_id)


if __name__ == "__main__":
    main()