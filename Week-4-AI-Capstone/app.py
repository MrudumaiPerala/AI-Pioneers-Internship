"""
Production-Ready Prediction API for Customer Churn Prediction
Built with FastAPI, Pydantic, and Joblib.
"""
import os
from pathlib import Path
from typing import List, Optional, Literal
from contextlib import asynccontextmanager

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path Configuration & Model State
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

model_pipeline = None

def load_model():
    global model_pipeline
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Please run the training script first."
        )
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model successfully loaded from {MODEL_PATH}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model pipeline
    try:
        load_model()
    except Exception as e:
        print(f"Warning on startup: {e}")
    yield
    # Shutdown logic if needed

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-End Machine Learning Capstone API for Telecom Customer Churn Prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas for Input Validation
# ---------------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., example="Female")
    SeniorCitizen: Literal[0, 1] = Field(..., example=0, description="1 if Senior Citizen, else 0")
    Partner: Literal["Yes", "No"] = Field(..., example="No")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=100, example=2, description="Number of months customer has stayed")
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., example="Fiber optic")
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(..., example="No")
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(..., example="No")
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(..., example="No")
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(..., example="No")
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(..., example="Yes")
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(..., example="Yes")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0.0, example=85.50, description="Monthly amount charged to the customer")
    TotalCharges: float = Field(..., ge=0.0, example=171.0, description="Total amount charged across tenure")

class BatchCustomerFeatures(BaseModel):
    customers: List[CustomerFeatures]

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for No Churn, 1 for Churn")
    churn: str = Field(..., description="'Yes' or 'No'")
    churn_probability: float = Field(..., description="Estimated probability of churning")
    risk_level: str = Field(..., description="'Low Risk', 'Moderate Risk', or 'High Risk'")
    status: str = "success"

class BatchPredictionResponse(BaseModel):
    total_customers: int
    predicted_churn_count: int
    predicted_churn_rate_pct: float
    results: List[PredictionResponse]

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "project": "AI Pioneers Internship - Week 4 AI Capstone",
        "title": "Customer Churn Prediction API",
        "status": "online",
        "model_loaded": model_pipeline is not None,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict_batch (POST)"
        }
    }

@app.get("/health", tags=["General"])
def health_check():
    if model_pipeline is None:
        try:
            load_model()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    return {"status": "healthy", "model": "loaded and ready"}

def calculate_risk_level(prob: float) -> str:
    if prob >= 0.65:
        return "High Risk"
    elif prob >= 0.35:
        return "Moderate Risk"
    return "Low Risk"

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerFeatures):
    if model_pipeline is None:
        load_model()
        
    try:
        # Convert input dictionary into DataFrame matching the pipeline training schema
        input_data = pd.DataFrame([customer.model_dump()])
        
        # Pipeline performs both pre-processing and model inference
        prediction_val = int(model_pipeline.predict(input_data)[0])
        prob_val = float(model_pipeline.predict_proba(input_data)[0][1])
        
        return PredictionResponse(
            prediction=prediction_val,
            churn="Yes" if prediction_val == 1 else "No",
            churn_probability=round(prob_val, 4),
            risk_level=calculate_risk_level(prob_val)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchCustomerFeatures):
    if model_pipeline is None:
        load_model()
        
    if not batch.customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list cannot be empty."
        )
        
    try:
        input_data = pd.DataFrame([c.model_dump() for c in batch.customers])
        predictions = model_pipeline.predict(input_data)
        probabilities = model_pipeline.predict_proba(input_data)[:, 1]
        
        results = []
        churn_count = 0
        for pred, prob in zip(predictions, probabilities):
            pred_int = int(pred)
            prob_flt = float(prob)
            if pred_int == 1:
                churn_count += 1
            results.append(PredictionResponse(
                prediction=pred_int,
                churn="Yes" if pred_int == 1 else "No",
                churn_probability=round(prob_flt, 4),
                risk_level=calculate_risk_level(prob_flt)
            ))
            
        return BatchPredictionResponse(
            total_customers=len(batch.customers),
            predicted_churn_count=churn_count,
            predicted_churn_rate_pct=round((churn_count / len(batch.customers)) * 100, 2),
            results=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )

# ---------------------------------------------------------------------------
# Direct execution support
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Customer Churn Prediction API on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
