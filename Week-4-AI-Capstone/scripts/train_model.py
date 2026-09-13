"""
Model Training, Evaluation & Serialization Script
Customer Churn Prediction - End-to-End Pipeline
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def run_training_pipeline():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "dataset.csv"
    model_dir = base_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Drop customer identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target variable conversion to binary 0/1
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

    # Coerce TotalCharges to numeric if needed
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numerical_features]

    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Preprocessing pipelines
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Candidate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = []
    trained_pipelines = {}

    print("\n" + "="*70)
    print("MODEL COMPARISON AND EVALUATION")
    print("="*70)

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest performing model selected: {best_model_name}")

    best_pipeline = trained_pipelines[best_model_name]

    # Save model artifact
    joblib.dump(best_pipeline, model_path)
    print(f"Successfully serialized end-to-end model pipeline to: {model_path}")

    # Verify reload
    loaded_pipe = joblib.load(model_path)
    sample_preds = loaded_pipe.predict(X_test.head(3))
    sample_probs = loaded_pipe.predict_proba(X_test.head(3))[:, 1]
    print("\nSanity Check on Serialized Model (First 3 test samples):")
    print(f"Predictions: {sample_preds}")
    print(f"Probabilities: {sample_probs.round(4)}")

    return results_df, best_model_name

if __name__ == "__main__":
    run_training_pipeline()
