# Week 4 AI Capstone: Telecom Customer Churn Prediction & REST API

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning capstone application that predicts customer churn risk in subscription telecommunications. Features a complete ML pipeline—from data ingestion and leak-free preprocessing to ensemble model training, hyperparameter optimization, model serialization with Joblib, and a production REST API built with FastAPI.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Architecture](#dataset-architecture)
4. [Technologies Used](#technologies-used)
5. [Data Preprocessing & Pipeline Architecture](#data-preprocessing--pipeline-architecture)
6. [Machine Learning Models & Evaluation](#machine-learning-models--evaluation)
7. [Project Directory Structure](#project-directory-structure)
8. [Installation & Setup](#installation--setup)
9. [Training & Model Serialization](#training--model-serialization)
10. [Running the Prediction API](#running-the-prediction-api)
11. [API Endpoints & Usage Examples](#api-endpoints--usage-examples)
12. [Presentation Deck](#presentation-deck)
13. [Conclusion & Business Impact](#conclusion--business-impact)

---

## Project Overview

In subscription businesses, retaining existing customers is significantly more cost-effective than customer acquisition. This project operationalizes machine learning by delivering:
- **`notebooks/model_development.ipynb`**: Complete exploratory data analysis, pipeline prototyping, model benchmarking, and export.
- **`data/dataset.csv`**: Representative customer dataset with 20 behavioral, contract, and demographic features.
- **`model/model.pkl`**: Serialized end-to-end Scikit-Learn `Pipeline` bundling preprocessing and the best estimator.
- **`app.py`**: Production-ready FastAPI application with interactive OpenAPI/Swagger docs, input validation via Pydantic, and batch inference.
- **`presentation/project_presentation.pptx`**: 10-slide executive presentation covering the entire solution.

---

## Problem Statement

Telecommunication providers face annual customer churn rates between 20% and 35%. Losing customers reduces recurring cash flow and increases sales pressure. Retention campaigns (discounts, service upgrades, loyalty perks) are expensive and can only be targeted effectively when high-risk accounts are identified with high precision before they terminate service.

**Objective**: Construct an end-to-end supervised machine learning system that ingests customer account characteristics and outputs both the churn classification and a calibrated probability score to triage accounts into **Low Risk**, **Moderate Risk**, or **High Risk** tiers.

---

## Dataset Architecture

The dataset (`data/dataset.csv`) contains 1,500 customer records with the following attributes:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `customerID` | Categorical | Unique customer identifier (dropped before modeling) |
| `gender` | Categorical | Male or Female |
| `SeniorCitizen` | Binary | 1 if senior citizen, 0 otherwise |
| `Partner` | Categorical | Whether customer has a partner (Yes/No) |
| `Dependents` | Categorical | Whether customer has dependents (Yes/No) |
| `tenure` | Numeric | Number of months customer has been with company (1–72) |
| `PhoneService` | Categorical | Whether customer has phone service (Yes/No) |
| `MultipleLines` | Categorical | Yes, No, or No phone service |
| `InternetService` | Categorical | DSL, Fiber optic, or No |
| `OnlineSecurity` | Categorical | Yes, No, or No internet service |
| `OnlineBackup` | Categorical | Yes, No, or No internet service |
| `DeviceProtection` | Categorical | Yes, No, or No internet service |
| `TechSupport` | Categorical | Yes, No, or No internet service |
| `StreamingTV` | Categorical | Yes, No, or No internet service |
| `StreamingMovies` | Categorical | Yes, No, or No internet service |
| `Contract` | Categorical | Month-to-month, One year, Two year |
| `PaperlessBilling` | Categorical | Yes or No |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| `MonthlyCharges` | Numeric | Monthly recurring charges in USD |
| `TotalCharges` | Numeric | Cumulative charges across customer tenure |
| **`Churn`** | **Target** | **Whether the customer churned (Yes / No)** |

---

## Technologies Used

- **Language**: Python 3.11+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (ColumnTransformer, Pipeline, GridSearchCV)
- **Model Serialization**: Joblib
- **Visualization**: Matplotlib, Seaborn
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **Presentation Deck**: Python-PPTX

---

## Data Preprocessing & Pipeline Architecture

To prevent **data leakage** between training and evaluation splits, and to ensure inference requires no external feature engineering scripts, all transformations are encapsulated in a single `ColumnTransformer`:

```
Raw Customer JSON / DataFrame
               │
   ┌───────────┴───────────┐
   ▼                       ▼
[Numerical Features]     [Categorical Features]
(tenure, MonthlyCharges,  (Contract, InternetService,
 TotalCharges)            PaymentMethod, etc.)
   │                       │
SimpleImputer(median)    SimpleImputer(most_frequent)
   │                       │
StandardScaler()         OneHotEncoder(handle_unknown='ignore')
   └───────────┬───────────┘
               ▼
       Concatenated Features
               ▼
     [RandomForestClassifier]
               ▼
  Prediction: 0 / 1 & Probability
```

---

## Machine Learning Models & Evaluation

Four candidate classification algorithms were benchmarked on an 80/20 stratified split:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **83.3%** | **78.4%** | **71.2%** | **0.746** | **0.865** |
| Gradient Boosting | 82.0% | 76.1% | 69.8% | 0.728 | 0.852 |
| Logistic Regression | 78.7% | 68.2% | 67.5% | 0.678 | 0.824 |
| Decision Tree | 76.3% | 64.9% | 61.2% | 0.630 | 0.751 |

**Selected Model**: **Random Forest Classifier** was chosen due to its balanced F1-score and highest ROC-AUC, providing reliable class probability calibration for risk stratification.

---

## Project Directory Structure

```text
Week-4-AI-Capstone/
│
├── data/
│   ├── dataset.csv                  # 1,500 sample telecom churn dataset
│   └── generate_data.py             # Script to regenerate dataset
│
├── notebooks/
│   └── model_development.ipynb      # Complete analysis, EDA, training & export
│
├── model/
│   └── model.pkl                    # Serialized end-to-end Pipeline artifact
│
├── presentation/
│   ├── project_presentation.pptx    # 10-slide capstone PowerPoint presentation
│   └── presentation_notes.md        # Comprehensive speaker notes & script
│
├── scripts/
│   ├── train_model.py               # Headless training and evaluation runner
│   ├── generate_notebook.py         # Programmatic notebook generator
│   └── generate_presentation.py     # Slide deck generator script
│
├── app.py                           # FastAPI prediction service
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## Installation & Setup

### 1. Clone the repository and navigate to the project directory:
```bash
cd Week-4-AI-Capstone
```

### 2. Activate virtual environment and install dependencies:
```bash
# Windows
..\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training & Model Serialization

To retrain the model and regenerate `model/model.pkl`:
```bash
python scripts/train_model.py
```

Output:
```text
Loading dataset from: .../data/dataset.csv
Dataset shape: (1500, 21)
MODEL COMPARISON AND EVALUATION
======================================================================
              Model  Accuracy  Precision   Recall  F1-Score  ROC-AUC
      Random Forest    0.8333     0.7838   0.7123    0.7462   0.8651
  Gradient Boosting    0.8200     0.7606   0.6986    0.7282   0.8519
Logistic Regression    0.7867     0.6824   0.6747    0.6784   0.8238
      Decision Tree    0.7633     0.6486   0.6123    0.6301   0.7510

Best performing model selected: Random Forest
Successfully serialized end-to-end model pipeline to: .../model/model.pkl
```

---

## Running the Prediction API

Start the FastAPI application with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- API Root: `http://127.0.0.1:8000/`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

---

## API Endpoints & Usage Examples

### 1. Health Check
- **GET** `/health`
```json
{
  "status": "healthy",
  "model": "loaded and ready"
}
```

### 2. Single Prediction (`POST /predict`)

#### Request (High Risk Customer):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

#### Response:
```json
{
  "prediction": 1,
  "churn": "Yes",
  "churn_probability": 0.7642,
  "risk_level": "High Risk",
  "status": "success"
}
```

---

#### Request (Loyal / Low Risk Customer):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Male",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "Yes",
       "tenure": 60,
       "PhoneService": "Yes",
       "MultipleLines": "Yes",
       "InternetService": "DSL",
       "OnlineSecurity": "Yes",
       "OnlineBackup": "Yes",
       "DeviceProtection": "Yes",
       "TechSupport": "Yes",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Two year",
       "PaperlessBilling": "No",
       "PaymentMethod": "Credit card (automatic)",
       "MonthlyCharges": 64.20,
       "TotalCharges": 3852.00
     }'
```

#### Response:
```json
{
  "prediction": 0,
  "churn": "No",
  "churn_probability": 0.0815,
  "risk_level": "Low Risk",
  "status": "success"
}
```

---

### 3. Batch Prediction (`POST /predict_batch`)
Send a JSON array containing up to hundreds of accounts under `"customers": [...]`. The API returns individual predictions along with aggregate metrics (`predicted_churn_count` and `predicted_churn_rate_pct`).

---

## Presentation Deck

The final presentation is located at:
- **`presentation/project_presentation.pptx`**: 10-slide 16:9 widescreen presentation.
- **`presentation/presentation_notes.md`**: Complete transcript and slide-by-slide speaker notes.

---

## Conclusion & Business Impact

This capstone delivers an enterprise-ready machine learning service that bridges the gap between notebook experimentation and real-time inference:
1. **Zero Preprocessing Mismatch**: Scikit-Learn `ColumnTransformer` guarantees identical transformations in both training and production.
2. **Actionable Confidence**: Probability scores allow businesses to prioritize intervention resources on the highest-risk customers first.
3. **Seamless Integration**: Standardized REST endpoints enable plug-and-play connection with CRM platforms, automated email pipelines, and business intelligence dashboards.
