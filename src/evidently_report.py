import pandas as pd
import os
import json

from src.logger import get_logger

logger = get_logger(__name__)


def generate_drift_report(reference_df, current_df):

    drift_summary = {}

    for col in reference_df.columns:

        if col not in current_df.columns:
            continue

        # Example logic (simplified)
        drift_summary[col] = {
            "type": "numerical",
            "severity": "LOW"
        }

    # ✅ Severity Count
    severity_count = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for v in drift_summary.values():
        severity = v.get("severity")
        if severity:
            severity_count[severity] += 1

    summary = {
        "total_columns": len(drift_summary),
        "severity": severity_count
    }

    return {
        "summary": summary,
        "details": drift_summary
    }