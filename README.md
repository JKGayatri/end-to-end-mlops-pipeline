# End-to-End MLOps Pipeline for House Price Prediction

## Overview
This project demonstrates a production-grade ML system with:
- Model training & inference separation
- FastAPI-based prediction service
- Batch prediction pipeline
- Data validation & schema enforcement
- Data drift detection with severity scoring
- Logging & monitoring

---

## Architecture

Train → Save Model → API → Batch Prediction → Drift Detection → Output CSV

---

## Tech Stack
- Python
- Scikit-learn
- FastAPI
- Pandas / NumPy
- Logging
- Custom Drift Detection

---

## Features

### Model Training
- Feature engineering pipeline
- Log transformation
- Model persistence using joblib

### Inference API
- Single prediction endpoint
- Batch prediction via CSV upload

### Data Validation
- Required vs optional column schema
- Input validation before inference

### Drift Detection
- Column-level drift detection
- Severity scoring (LOW / MEDIUM / HIGH)

### Batch Pipeline
- Upload CSV → get predictions + drift report

---

## API Usage

### Batch Prediction

POST `/predict-batch`

Upload CSV → returns:
- Predictions file
- Drift summary

---

## Output

- predictions/test_predictions.csv

---

## Key Learnings
- Production ML pipelines
- MLOps fundamentals
- API-based deployment
- Data drift monitoring

---

## Future Improvements
- CI/CD pipeline
- Dockerization
- Model versioning
- Real-time monitoring dashboard
