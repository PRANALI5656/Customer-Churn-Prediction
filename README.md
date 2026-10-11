# Customer Churn Prediction

A trained XGBoost model and FastAPI application for predicting customer churn.

## Project layout

```text
main.py                         FastAPI application
src/model.py                    Feature preprocessing, training, artifact loading
scripts/train_model.py          Reproducible model training command
artifacts/churn_xgb.joblib       Saved preprocessing and classifier pipeline
artifacts/metadata.json          Threshold, feature schema, and holdout metrics
data/raw/                        Canonical training CSV
data/external/                   Preserved copy of the original supplied CSV
notebooks/EDA.ipynb              Exploratory analysis and model experiments
requirements.txt                 API and model dependencies
requirements-notebooks.txt      Optional EDA and experiment dependencies
Dockerfile                       Container image for the API
```

## Run the API locally

Use Python 3.13 to match the saved model artifact. From the project root, install the API dependencies
and start the server:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/docs> for interactive API documentation.
`GET /health` checks that the model loaded, and `POST /predict` accepts
the raw customer fields (excluding `customerID` and `Churn`) and returns a
churn label, probability, and decision threshold.

The model was tuned to favor churn recall; its 0.3 threshold is a business
trade-off that should be evaluated for the intended retention workflow.

## Retrain the model

The saved pipeline includes the same preprocessing steps used at prediction
time, so training and inference use a consistent feature schema:

```powershell
python -m scripts.train_model
```

This reads `data/raw/Telco-Customer-Churn.csv`, reproduces the notebook's
80/20 stratified split and selected XGBoost parameters, then writes the model
and holdout metrics to `artifacts/`. Pass `--data` or `--artifacts` to use
different paths. Install `requirements-notebooks.txt` only when running the
EDA and optional LightGBM, Optuna, or MLflow experiments.

## Run tests

```powershell
python -m pip install pytest httpx2
python -m pytest
```

## Build the container

```powershell
docker build -t customer-churn-api .
docker run --rm -p 8000:10000 customer-churn-api
```

Open `http://localhost:8000/docs` in your browser. The container listens on
port `10000`; `-p 8000:10000` maps your computer's port `8000` to it. Uvicorn
may print `0.0.0.0:10000` in the terminal; `0.0.0.0` is the container's bind
address, not the address to enter in your browser. Use `localhost:8000`.
If a hosting platform sets `PORT`, the container uses that port instead, so
map your chosen host port to that container port (for example,
`-e PORT=10000 -p 8000:10000`).
