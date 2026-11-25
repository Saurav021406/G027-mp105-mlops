# G027 - MP105: Parcel Delivery Days (Bhopal)

- Group id: G027
- Problem id: MP105
- Faculty: Amit Soni, Batch 69
- Client: Bhopal Speed Parcel Hub (predict `delivery_days`, metric MAE)

## Members

| Enrolment no. | Name | GitHub |
|---|---|---|
| S24CSEU2606 | Urjita Jain | Urjitaa |
| S24CSEU2576 | Kumar Saurav | Saurav021406 |
| S24CSEU0104 | Manan Garg | Manan7289 |
| S24CSEU2605 | Lakshita Badlia | lakshitabadlia |
| S24CSEU2604 | Ruhi Mishra | RuhiMishra2202 |

## Run

    pip install -r requirements.txt
    set PYTHONPATH=src;.        (Linux/Mac: export PYTHONPATH=src:.)
    python -m pytest tests -v
    python -m parcel_days.train      # v1 with MLflow tracking
    python -m parcel_days.retrain    # v2 and promotion rule
    python -m parcel_days.predict    # holdout predictions
    uvicorn app.main:app --port 8000

## Docker and Kubernetes

    docker build -t mp105-api:1.0 .
    docker run -p 8000:8000 mp105-api:1.0
    kubectl apply -f k8s/

## Data

`train.csv` and `later.csv` are tracked with DVC (see `data/raw/*.dvc`).
