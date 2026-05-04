#src/validation.py
from src.schema import REQUIRED_COLUMNS

def validate_required_columns(df):
    input_cols = set(df.columns)
    missing = [col for col in REQUIRED_COLUMNS if col not in input_cols]

    if missing:
        raise ValueError(f"Missing REQUIRED columns: {missing}")