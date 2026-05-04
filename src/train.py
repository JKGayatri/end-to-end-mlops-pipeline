# src/train.py

import os
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.logger import get_logger

logger = get_logger(__name__)

# -------------------------------
# 1. Load Data
# -------------------------------
logger.info("Loading training data...")
df = load_data("data/train.csv")

# -------------------------------
# 2. Feature Engineering
# -------------------------------
logger.info("Applying feature engineering...")
df = create_features(df)



# -------------------------------
# 3. Target Transformation
# -------------------------------
df['SalePrice'] = np.log(df['SalePrice'])

# -------------------------------
# 4. Split X, y
# -------------------------------
logger.info("Splitting dataset...")
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Save schema
schema = {
    "version": "1.0",
    "columns": X.columns.tolist(),
    "required_columns": ["OverallQual", "GrLivArea"],
    "created_at": str(datetime.now())
}

with open("models/schema.json", "w") as f:
    json.dump(schema, f, indent=4)

logger.info("Schema saved!")

# Save reference dataset for drift comparison
X.to_csv("models/reference_data.csv", index=False)

logger.info("Reference data saved!")

# -------------------------------
# 5. Identify Columns
# -------------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
stats = {}

for col in num_cols:
    stats[col] = {
        "mean": float(X[col].mean()),
        "std": float(X[col].std())
    }

with open("models/data_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

logger.info("Training stats saved!")
cat_cols = X.select_dtypes(include=['object']).columns

# -------------------------------
# 6. Create Pipelines
# -------------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# -------------------------------
# 7. Full Pipeline
# -------------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor())
])

# -------------------------------
# 8. GridSearchCV
# -------------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# -------------------------------
# 10. Evaluate on Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# 9. Train Model
# -------------------------------
try:
    logger.info("Starting model training...")
    grid_search.fit(X, y)
    logger.info("Training completed")

except Exception as e:
    logger.error(f"Error occurred during model training: {e}", exc_info=True)
    raise e

logger.info(f"Best Params: {grid_search.best_params_}")
logger.info(f"Best CV Score: {grid_search.best_score_}")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


logger.info(f"Final RMSE: {rmse}")
logger.info(f"Final R2: {r2}")

# -------------------------------
# 11. Save Model
# -------------------------------
logger.info("Saving model...")
joblib.dump(best_model, "models/pipeline.pkl")
joblib.dump(X.columns.tolist(), "models/columns.pkl")

logger.info("Model saved successfully!")

# # -------------------------------
# # 12. Load External Test Data
# # -------------------------------
# test_df = load_data("data/test.csv")

# # Apply feature engineering
# test_df = create_features(test_df)

# # Split features
# if "SalePrice" in test_df.columns:
#     X_test_ext = test_df.drop("SalePrice", axis=1)
#     y_test_ext = np.log(test_df["SalePrice"])
# else:
#     X_test_ext = test_df.copy()
#     y_test_ext = None

# # Align columns
# train_cols = X.columns.tolist()

# for col in train_cols:
#     if col not in X_test_ext:
#         X_test_ext[col] = None

# X_test_ext = X_test_ext[train_cols]

# # -------------------------------
# # 13. Predict
# # -------------------------------
# y_pred_ext = best_model.predict(X_test_ext)
# y_pred_ext = np.exp(y_pred_ext)

# # -------------------------------
# # 14. Evaluate (if possible)
# # -------------------------------
# if y_test_ext is not None:
#     y_test_ext = np.exp(y_test_ext)

#     rmse = np.sqrt(mean_squared_error(y_test_ext, y_pred_ext))
#     r2 = r2_score(y_test_ext, y_pred_ext)

    
#     logger.info(f"External Test RMSE: {rmse}")
#     logger.info(f"External Test R2: {r2}")

# # -------------------------------
# # 15. SAVE PREDICTIONS  ✅ HERE
# # -------------------------------
# output = X_test_ext.copy()
# output["PredictedPrice"] = y_pred_ext

# # Ensure folder exists
# os.makedirs("predictions", exist_ok=True)

# output.to_csv("predictions/test_predictions.csv", index=False)

# logger.info("Predictions saved!")