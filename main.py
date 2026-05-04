# main.py

from fastapi import FastAPI, UploadFile, File
import joblib
from pydantic import BaseModel
from fastapi.responses import FileResponse
import uuid
import shutil
import os

from src.predict import predict
from src.logger import get_logger
from src.batch_predict import run_batch_prediction

logger = get_logger(__name__)

app = FastAPI(title="House Price Prediction API",
    description="ML API for real-time and batch predictions",
    version="1.0")


# -------------------------------
# Input Schema 
# -------------------------------
class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float


# -------------------------------
# Root API
# -------------------------------
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}


# -------------------------------
# Prediction API
# -------------------------------
@app.post("/predict")
def predict_price(data: HouseData):
    try:
        logger.info(f"Received data for prediction: {data}")
        prediction = predict(data.model_dump(exclude_none=True))
        logger.info(f"Prediction successful: {prediction}")
        return {"predicted_price": prediction}
    except Exception as e:
        logger.error(f"Error occurred while making prediction: {e}", exc_info=True)
        return {"error": "Prediction failed. Please check the input data and try again."}
    
# -------------------------------
# Batch Prediction API
# -------------------------------
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):

    try:
        logger.info(f"Received batch file: {file.filename}")

        # VALIDATION for file type and extension
        if not file.filename or not file.filename.endswith(".csv"):
            return {"error": "Invalid file. Please upload a CSV file."}
        if file.content_type != "text/csv":
            return {"error": "Invalid file type. Only CSV allowed."}
        
        # Save uploaded file temporarily
        temp_input = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)

        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate a unique output path
        os.makedirs("predictions", exist_ok=True)
        output_path = f"predictions/output_{uuid.uuid4().hex}.csv"

        # -------------------------------
        # Load model
        # -------------------------------
        logger.info("Loading model from models/pipeline.pkl")
        model = joblib.load("models/pipeline.pkl")
        columns = joblib.load("models/columns.pkl")
        
        result = run_batch_prediction(temp_input, output_path, model, columns)
        os.remove(temp_input)

        return {
            "message": "Batch prediction completed",
            "drift_summary": result["drift"]["summary"],
            "report_url": "drift-report",
            "file": result["predictions_file"],
            "download_url": "/download/predictions"
            }
    
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return {"error": str(ve)}

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        return {"error": "Batch prediction failed"}
    
# -------------------------------
# Download API for predictions
# -------------------------------
@app.get("/download/predictions")
def download_predictions():
    return FileResponse(
        path="predictions/output.csv",
        filename="predictions.csv",
        media_type="text/csv"
    )

# -------------------------------
# Download API for drift report
# -------------------------------
@app.get("/drift-report")
def get_drift_report():
    return FileResponse(
        path="reports/drift_report.html",
        media_type="text/html"
    )
