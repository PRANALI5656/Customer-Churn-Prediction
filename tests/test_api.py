from fastapi.testclient import TestClient

from main import app


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.5,
    "TotalCharges": 650.25,
}


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Yes", "No"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["threshold"] == 0.3
    assert body["model"] == "XGBClassifier"


def test_predict_rejects_unknown_categories() -> None:
    customer = {**VALID_CUSTOMER, "InternetService": "Satellite"}
    with TestClient(app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_accepts_missing_total_charges() -> None:
    customer = {**VALID_CUSTOMER, "TotalCharges": None}
    with TestClient(app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 200
