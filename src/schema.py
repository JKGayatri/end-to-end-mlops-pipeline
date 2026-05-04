# src/schema.py

# Columns that MUST be present (model will break without them)
REQUIRED_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF"
]

# Columns that can be missing (will be auto-filled)
OPTIONAL_COLUMNS = [
    # example
    "GarageArea",
    "YearBuilt",
    "LotFrontage"
]