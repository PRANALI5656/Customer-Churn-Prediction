from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILENAME = "churn_xgb.joblib"
METADATA_FILENAME = "metadata.json"
DECISION_THRESHOLD = 0.3
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
BINARY_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]
CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
BINARY_CATEGORIES = [
    ["Female", "Male"],
    ["No", "Yes"],
    ["No", "Yes"],
    ["No", "Yes"],
    ["No", "Yes"],
]
TARGET = "Churn"
MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": 511,
    "learning_rate": 0.011365625326696651,
    "max_depth": 3,
    "subsample": 0.798858904894541,
    "colsample_bytree": 0.6155803534220775,
    "min_child_weight": 7,
    "gamma": 2.6442162483006606,
    "reg_alpha": 4.3309785867578485,
    "reg_lambda": 4.138960193479452,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Build the fitted preprocessing and classification steps as one artifact."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "binary",
                OrdinalEncoder(categories=BINARY_CATEGORIES),
                BINARY_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = XGBClassifier(
        **MODEL_PARAMETERS,
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", model),
        ]
    )


def load_artifacts(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> tuple[Pipeline, dict[str, Any]]:
    model_path = artifact_dir / MODEL_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    missing = [path.name for path in (model_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Model artifacts are missing from {artifact_dir}: {', '.join(missing)}. "
            "Run `python scripts/train_model.py` from the project root first."
        )

    try:
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Could not load model artifacts from {artifact_dir}: {exc}") from exc
    if not isinstance(model, Pipeline) or not isinstance(metadata, dict):
        raise TypeError(f"Model artifacts in {artifact_dir} have an unexpected format.")
    return model, metadata


def train_and_save(
    data_path: Path = DEFAULT_DATA_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    data = pd.read_csv(data_path)
    required_columns = set(FEATURES + [TARGET, "customerID"])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")

    features = data[FEATURES].copy()
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")
    target = data[TARGET].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("The Churn column must contain only 'No' and 'Yes'.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(x_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "churn_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "churn_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "churn_f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / MODEL_FILENAME)
    metadata: dict[str, Any] = {
        "model": "XGBClassifier",
        "target": TARGET,
        "threshold": DECISION_THRESHOLD,
        "positive_class": "Yes",
        "feature_columns": FEATURES,
        "training_data": data_path.name,
        "training_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "row_count": int(len(data)),
        "python_version": sys.version.split()[0],
        "package_versions": {
            name: version(name)
            for name in ["fastapi", "joblib", "numpy", "pandas", "scikit-learn", "xgboost"]
        },
        "train_test_split": {"test_size": 0.2, "random_state": RANDOM_STATE},
        "model_parameters": {
            **MODEL_PARAMETERS,
            "scale_pos_weight": scale_pos_weight,
        },
        "holdout_metrics": metrics,
    }
    (artifact_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
