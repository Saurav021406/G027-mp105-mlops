from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD = {
    "distance_km": 1200, "weight_kg": 10.5, "hub_backlog": 2500,
    "pin_code_tier": 2, "declared_value_rs": 8000, "transfers": 3,
    "service": "express", "destination": "Metro",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_predict_rejects_distance_over_4500():
    r = client.post("/predict", json={**GOOD, "distance_km": 5000})
    assert r.status_code == 422


def test_predict_rejects_negative_transfers():
    r = client.post("/predict", json={**GOOD, "transfers": -1})
    assert r.status_code == 422