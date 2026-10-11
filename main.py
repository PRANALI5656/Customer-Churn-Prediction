from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.model import DECISION_THRESHOLD, load_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, metadata = load_artifacts()
    app.state.model = model
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn using the trained XGBoost model.",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    gender: Literal["Female", "Male"]
    senior_citizen: Annotated[int, Field(alias="SeniorCitizen", ge=0, le=1)]
    partner: Literal["No", "Yes"] = Field(alias="Partner")
    dependents: Literal["No", "Yes"] = Field(alias="Dependents")
    tenure: Annotated[int, Field(ge=0)]
    phone_service: Literal["No", "Yes"] = Field(alias="PhoneService")
    multiple_lines: Literal["No", "No phone service", "Yes"] = Field(alias="MultipleLines")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(alias="InternetService")
    online_security: Literal["No", "No internet service", "Yes"] = Field(alias="OnlineSecurity")
    online_backup: Literal["No", "No internet service", "Yes"] = Field(alias="OnlineBackup")
    device_protection: Literal["No", "No internet service", "Yes"] = Field(alias="DeviceProtection")
    tech_support: Literal["No", "No internet service", "Yes"] = Field(alias="TechSupport")
    streaming_tv: Literal["No", "No internet service", "Yes"] = Field(alias="StreamingTV")
    streaming_movies: Literal["No", "No internet service", "Yes"] = Field(alias="StreamingMovies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(alias="Contract")
    paperless_billing: Literal["No", "Yes"] = Field(alias="PaperlessBilling")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(alias="PaymentMethod")
    monthly_charges: Annotated[float, Field(alias="MonthlyCharges", ge=0)]
    total_charges: Annotated[float | None, Field(alias="TotalCharges")] = None


class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float
    threshold: float
    model: str


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="The model has not been loaded.")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    metadata = getattr(request.app.state, "metadata", None)
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="The model has not been loaded.")

    features = customer.model_dump(by_alias=True, mode="python")
    try:
        probability = float(model.predict_proba(pd.DataFrame([features]))[0][1])
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    threshold = float(metadata.get("threshold", DECISION_THRESHOLD))
    model_name = str(metadata.get("model", "XGBClassifier"))
    return PredictionResponse(
        prediction="Yes" if probability >= threshold else "No",
        churn_probability=probability,
        threshold=threshold,
        model=model_name,
    )
