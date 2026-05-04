import json
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

# Load stats once
with open("models/data_stats.json", "r") as f:
    TRAIN_STATS = json.load(f)


def detect_drift(df, threshold=3):
    drift_report = {}

    for col, stats in TRAIN_STATS.items():
        if col not in df.columns:
            continue

        mean = df[col].mean()

        train_mean = stats["mean"]
        train_std = stats["std"]

        if train_std == 0:
            continue

        z_score = abs((mean - train_mean) / train_std)

        if z_score > threshold:
            drift_report[col] = {
                "z_score": z_score,
                "status": "DRIFT DETECTED"
            }

    return drift_report