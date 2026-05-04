# src/predict.py

import joblib
import json
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.schema import REQUIRED_COLUMNS

logger = get_logger(__name__)

with open("models/schema.json", "r") as f:
    schema = json.load(f)

EXPECTED_COLUMNS = schema["columns"]
REQUIRED_COLUMNS = schema["required_columns"]
SCHEMA_VERSION = schema["version"]

# -------------------------------
# Load once (IMPORTANT)
# -------------------------------
logger.info("Loading model at startup...")
model = joblib.load("models/pipeline.pkl")
train_cols = joblib.load("models/columns.pkl")


# -------------------------------
# ✅ Validation
# -------------------------------
def validate_input(data: dict):
    missing = [col for col in REQUIRED_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing REQUIRED fields: {missing}")


# -------------------------------
# ✅ Predict function
# -------------------------------
def predict(data: dict):

    logger.info("🔥 NEW PREDICT FUNCTION CALLED")

    # -------------------------------
    # Step 1: Validate REQUIRED fields
    # -------------------------------
    validate_input(data)

    df = pd.DataFrame([data])

    logger.info(f"Input columns: {df.columns.tolist()}")

    # -------------------------------
    # Step 2: Add OPTIONAL missing cols
    # -------------------------------
    missing_cols = []
    for col in train_cols:
        if col not in df.columns:
            df[col] = None   # ✅ FIXED
            missing_cols.append(col)

    logger.info(f"Missing columns handled: {len(missing_cols)}")

    # -------------------------------
    # Step 3: Ensure column order
    # -------------------------------
    df = df[train_cols]

    # -------------------------------
    # Step 4: Predict
    # -------------------------------
    prediction = model.predict(df)[0]

    real_price = float(np.exp(prediction))

    return real_price