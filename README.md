# AI Pioneers Internship - Machine Learning Portfolio

A comprehensive, 4-week Machine Learning internship portfolio encompassing Python preprocessing, supervised learning, unsupervised learning, model evaluation, serialization, and production REST API deployment.

---

## Repository Structure

```text
AI-Pioneers-Internship/
│
├── Week-1-Python-ML-Preprocessing/
│   ├── preprocessing.ipynb          # Missing values, encoding, scaling, EDA
│   ├── Week1.py                     # Standalone preprocessing script
│   ├── README.md                    # Week 1 documentation
│   └── data/
│       ├── sample_dataset.csv       # Raw input dataset
│       └── cleaned_dataset.csv      # Processed dataset
│
├── Week-2-Supervised-ML/
│   ├── supervised_ml.ipynb          # Logistic Regression, Decision Tree, Random Forest, KNN
│   ├── README.md                    # Week 2 documentation
│   └── data/
│
├── Week-3-Unsupervised-ML/
│   ├── unsupervised_evaluation.ipynb # K-Means, PCA, Hierarchical clustering, GridSearchCV
│   ├── README.md                    # Week 3 documentation
│   └── data/
│
└── Week-4-AI-Capstone/
    ├── data/
    │   ├── dataset.csv              # 1,500 record Telecom Customer Churn dataset
    │   └── generate_data.py         # Synthetic dataset generator
    │
    ├── notebooks/
    │   └── model_development.ipynb  # End-to-end EDA, model comparison & serialization
    │
    ├── model/
    │   └── model.pkl                # Serialized Scikit-Learn Pipeline
    │
    ├── presentation/
    │   ├── project_presentation.pptx # 10-slide capstone presentation deck
    │   └── presentation_notes.md    # Complete slide-by-slide speaker notes
    │
    ├── scripts/
    │   ├── train_model.py           # Headless pipeline trainer & evaluator
    │   ├── generate_notebook.py     # Programmatic notebook generator
    │   └── generate_presentation.py # Presentation builder
    │
    ├── app.py                       # FastAPI Prediction REST API
    ├── requirements.txt             # Capstone dependencies
    └── README.md                    # Full Capstone project documentation
```

---

## Weekly Curriculum Overview

### [Week 1: Python ML Preprocessing](Week-1-Python-ML-Preprocessing/README.md)
- **Focus**: Data cleaning, imputation strategies, and normalization.
- **Key Techniques**:
  - Imputation (Median for numerical, Mode for categorical).
  - Duplicate detection and removal.
  - Categorical dummy encoding (`pd.get_dummies`).
  - Feature normalization with Z-Score scaling.
  - Exploratory statistical summaries and correlation analysis.

### [Week 2: Supervised Machine Learning](Week-2-Supervised-ML/README.md)
- **Focus**: Comparing supervised classification algorithms on tabular benchmarks.
- **Key Algorithms**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
- **Metrics**: Accuracy, Precision, Recall, and F1-Score.

### [Week 3: Unsupervised Learning & Model Evaluation](Week-3-Unsupervised-ML/README.md)
- **Focus**: Unsupervised pattern discovery, clustering, and evaluation.
- **Key Techniques**:
  - K-Means Clustering & Elbow Method.
  - Hierarchical Clustering with Dendrograms.
  - Principal Component Analysis (PCA) Dimensionality Reduction.
  - Silhouette Score analysis.
  - Confusion Matrices, ROC-AUC, K-Fold Cross-Validation, and Hyperparameter Tuning with `GridSearchCV`.

### [Week 4: AI Project Deployment & Capstone](Week-4-AI-Capstone/README.md)
- **Focus**: **Flagship End-to-End ML Application & Deployment**.
- **Project**: **Telecom Customer Churn Prediction**.
- **Deliverables**:
  - End-to-end `ColumnTransformer` + `Pipeline` preventing data leakage.
  - Model serialization using `joblib` into `model.pkl`.
  - Production REST API built with **FastAPI** featuring interactive Swagger docs (`/docs`), single inference (`/predict`), and batch inference (`/predict_batch`).
  - 10-slide executive presentation (`project_presentation.pptx`) with speaker notes.
  - Complete project documentation and verification tests.

---

## Quickstart: Running the Capstone API

1. **Activate the Virtual Environment**:
   ```bash
   .\.venv\Scripts\activate
   ```

2. **Navigate to Week 4 and Install Requirements**:
   ```bash
   cd Week-4-AI-Capstone
   pip install -r requirements.txt
   ```

3. **Train and Serialize the Model**:
   ```bash
   python scripts/train_model.py
   ```

4. **Launch the FastAPI Server**:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Test in Browser**:
   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the interactive API.
