import json
import logging
import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = ["distance_km", "weight_kg", "hub_backlog", "pin_code_tier",
            "declared_value_rs", "transfers", "service", "destination"]


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"level": record.levelname, "message": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("mp105-api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False

app = FastAPI(title="MP105 Parcel Delivery Days")
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    distance_km: float = Field(..., le=4500)   # client rule 1
    weight_kg: float
    hub_backlog: int
    pin_code_tier: int
    declared_value_rs: float
    transfers: int = Field(..., ge=0)          # client rule 2
    service: Literal["standard", "express", "economy"]
    destination: Literal["Metro", "Tier-2", "Rural"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    prediction = float(model.predict(row)[0])
    logger.info("prediction served", extra={"extra_fields": {
        "prediction": round(prediction, 3),
        "service": req.service,
        "destination": req.destination,
    }})
    return {"prediction": prediction}