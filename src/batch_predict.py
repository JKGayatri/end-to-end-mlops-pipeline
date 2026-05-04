# batch_predict.py
import json
import numpy as np
import os
import pandas as pd
import joblib

from src.logger import get_logger
from src.data_loader import load_data
from src.feature_engineering import create_features
from src.validation import validate_required_columns
from src.drift import detect_drift
from src.evidently_report import generate_drift_report

logger = get_logger(__name__)

with open("models/schema.json", "r") as f:
    schema = json.load(f)

EXPECTED_COLUMNS = schema["columns"]
REQUIRED_COLUMNS = schema["required_columns"]
SCHEMA_VERSION = schema["version"]

def run_batch_prediction(input_path: str, output_path: str, model, columns):
    
    # -------------------------------
    # 1. Load test data
    # -------------------------------
    logger.info("Loading test data...")
    test_df = load_data(input_path)

    reference_df = pd.read_csv("models/reference_data.csv")

    # -------------------------------
    # 2: Validate REQUIRED columns
    # -------------------------------
    validate_required_columns(test_df)

    # Apply feature engineering
    logger.info("Applying feature engineering...")
    test_df = create_features(test_df)

    #Evidently report
    reference_df = pd.read_csv("models/reference_data.csv")

    drift_report = generate_drift_report(
    reference_df=reference_df,
    current_df=test_df
    )

    # Optional: keep lightweight drift signal
    simple_drift = detect_drift(test_df)

    if simple_drift:
        logger.warning(f"Data drift detected: {simple_drift}")

    # -------------------------------
    # 3. Prepare input
    # -------------------------------
    if "SalePrice" in test_df.columns:
        X_test = test_df.drop("SalePrice", axis=1)
    else:
        X_test = test_df.copy()

    # Align columns
    for col in columns:
        if col not in X_test:
            X_test[col] = None

    X_test = X_test[columns]

    # -------------------------------
    # 4. Predict
    # -------------------------------
    logger.info("Running predictions...")
    y_pred = model.predict(X_test)
    y_pred = np.exp(y_pred)

    # -------------------------------
    # 5. Save predictions
    # -------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = test_df.copy()
    output["PredictedPrice"] = y_pred
    logger.info("Saving predictions...")
    output.to_csv(output_path, index=False)
    logger.info("Batch prediction completed successfully!")

    # Return drift info and report path
    return {
        "drift": drift_report,
        "predictions_file": output_path
    }

    

# CLI execution
if __name__ == "__main__":
    run_batch_prediction(
        input_path="data/test.csv",
        output_path="predictions/test_predictions.csv"
    )