from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.churn_pipeline import RAW_FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
app = Flask(__name__)


def validate_customer_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    missing_fields = [field for field in RAW_FEATURE_COLUMNS if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    customer = {field: payload[field] for field in RAW_FEATURE_COLUMNS}
    numeric_fields = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
    for field in numeric_fields:
        value = customer[field]
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"Field '{field}' must be numeric")
        try:
            float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Field '{field}' must be numeric") from error
    for field, value in customer.items():
        if field not in numeric_fields and not isinstance(value, str):
            raise ValueError(f"Field '{field}' must be a string")
    return customer


@app.get("/health")
def health() -> tuple[object, int]:
    status = "ready" if model is not None else "model_missing"
    return jsonify({"status": status}), 200 if model is not None else 503


@app.post("/predict")
def predict() -> tuple[object, int]:
    if model is None:
        return jsonify({"error": "Model artifact is not available"}), 503
    try:
        customer = validate_customer_payload(request.get_json(silent=True))
        customer_frame = pd.DataFrame([customer])
        prediction = int(model.predict(customer_frame)[0])
        probability = float(model.predict_proba(customer_frame)[0][1])
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(
        {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 6),
        }
    ), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
