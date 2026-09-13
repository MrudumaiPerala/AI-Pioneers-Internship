"""
End-to-End Verification Suite for Week 4 AI Capstone
Tests pipeline loading, single prediction, batch prediction, and API responses.
"""
import sys
from pathlib import Path
import pandas as pd
import joblib

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

def test_model_artifact():
    print("\n[TEST 1] Verifying Serialized Model Artifact...")
    model_path = base_dir / "model" / "model.pkl"
    assert model_path.exists(), f"Model file missing at {model_path}"
    
    pipeline = joblib.load(model_path)
    assert pipeline is not None, "Pipeline failed to load"
    print(f"PASS: Successfully loaded model pipeline from {model_path}")
    return pipeline

def test_inference(pipeline):
    print("\n[TEST 2] Testing Single Inference Pipeline...")
    # Sample high churn risk customer
    sample_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.00
    }
    
    df_sample = pd.DataFrame([sample_data])
    pred = pipeline.predict(df_sample)[0]
    prob = pipeline.predict_proba(df_sample)[0][1]
    
    print(f"Inference Result: Prediction={pred}, Churn Probability={prob:.4f}")
    assert pred in [0, 1], f"Invalid prediction value {pred}"
    assert 0.0 <= prob <= 1.0, f"Invalid probability {prob}"
    print("PASS: Single inference executed with valid bounds.")

def test_fastapi_endpoints():
    print("\n[TEST 3] Testing FastAPI Endpoints...")
    from app import app, root, health_check, predict, predict_batch, CustomerFeatures, BatchCustomerFeatures, load_model
    
    load_model()
    
    # 1. Root
    res_root = root()
    assert res_root["status"] == "online", f"Root status not online: {res_root}"
    print("PASS: GET / endpoint returned online status")
    
    # 2. Health
    res_health = health_check()
    assert res_health["status"] == "healthy", f"Health status unhealthy: {res_health}"
    print("PASS: GET /health endpoint returned healthy status")
    
    # 3. Predict Single
    payload_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.00
    }
    customer_obj = CustomerFeatures(**payload_data)
    res_pred = predict(customer_obj)
    assert res_pred.prediction in [0, 1]
    assert res_pred.churn in ["Yes", "No"]
    assert 0.0 <= res_pred.churn_probability <= 1.0
    print(f"PASS: POST /predict endpoint returned: {res_pred.model_dump()}")
    
    # 4. Predict Batch
    batch_obj = BatchCustomerFeatures(customers=[customer_obj, customer_obj])
    res_batch = predict_batch(batch_obj)
    assert res_batch.total_customers == 2
    assert len(res_batch.results) == 2
    print(f"PASS: POST /predict_batch endpoint returned {res_batch.total_customers} predictions")

def run_all_tests():
    print("="*60)
    print("RUNNING CAPSTONE VALIDATION SUITE")
    print("="*60)
    pipe = test_model_artifact()
    test_inference(pipe)
    test_fastapi_endpoints()
    print("\nALL SUITE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
