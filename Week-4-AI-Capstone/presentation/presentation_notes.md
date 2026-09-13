# Week 4 AI Capstone Presentation Notes & Speaker Script

**Project Title**: Telecom Customer Churn Prediction & REST API Deployment  
**Internship**: AI Pioneers Internship — Final Capstone  
**Presenter**: ML Engineer Intern  

---

## Slide 1: Title & Capstone Overview
- **Header**: AI Capstone | Slide 01
- **Title**: Customer Churn Prediction: End-to-End Machine Learning Pipeline & Deployment API
- **Speaker Notes**:
  > *"Good morning/afternoon everyone. Today, I am excited to present my final AI Capstone project for the AI Pioneers Internship: an end-to-end Machine Learning pipeline and live prediction API for Customer Churn Prediction. Over the past 4 weeks, we covered Python data preprocessing, supervised learning, and unsupervised clustering. Today, I've brought all of these disciplines together into a production-grade, serialized machine learning service."*

---

## Slide 2: Problem Statement
- **Header**: AI Capstone | Slide 02
- **Title**: Problem Statement — Why Churn Prediction Matters
- **Speaker Notes**:
  > *"In subscription businesses, customer retention is paramount. Acquiring a new customer costs 5 to 7 times more than retaining an existing one. When customers churn silently, the business incurs compounding revenue loss. Our challenge was: how can telecom operators proactively identify which customers are at risk of leaving, well before they terminate their contracts?"*

---

## Slide 3: Project Objectives
- **Header**: AI Capstone | Slide 03
- **Title**: Project Objectives — End-to-End Technical Milestones
- **Speaker Notes**:
  > *"To solve this problem, we established five key technical milestones:
  > 1. Ingest and explore customer demographic and service usage data.
  > 2. Build a leak-free preprocessing pipeline using Scikit-Learn's ColumnTransformer.
  > 3. Train and benchmark 4 different classification algorithms.
  > 4. Serialize the optimal model pipeline with Joblib.
  > 5. Expose the serialized model through a production-ready FastAPI REST service with interactive Swagger documentation."*

---

## Slide 4: Dataset Architecture & Exploratory Data Analysis
- **Header**: AI Capstone | Slide 04
- **Title**: Dataset Architecture & Key EDA Insights
- **Speaker Notes**:
  > *"We analyzed a dataset of 1,500 customer records featuring 20 distinct attributes spanning tenure, monthly charges, contract terms, and technical services. Our EDA revealed a baseline churn rate of approximately 26%. Crucially, we discovered strong behavioral signals: customers on month-to-month contracts and those using fiber optic internet without tech support or online security displayed dramatically higher attrition rates compared to two-year contract holders."*

---

## Slide 5: Data Preprocessing Pipeline
- **Header**: AI Capstone | Slide 05
- **Title**: Preprocessing & Feature Engineering Architecture
- **Speaker Notes**:
  > *"A common mistake in ML projects is training-serving skew: where preprocessing in production differs from training. To eliminate this risk, we designed a unified Scikit-Learn ColumnTransformer. Numerical features (tenure, monthly charges) undergo median imputation and standard scaling. Categorical features undergo mode imputation and One-Hot Encoding with unknown value handling. This entire transformer is bundled directly with our classifier, meaning our deployment API accepts raw JSON records without needing any ad-hoc data manipulation."*

---

## Slide 6: Supervised Model Selection
- **Header**: AI Capstone | Slide 06
- **Title**: Supervised Model Selection & Benchmark Comparison
- **Speaker Notes**:
  > *"We benchmarked four distinct model families using an 80/20 stratified split:
  > - Logistic Regression as an interpretable linear baseline (~78% accuracy, 0.82 ROC-AUC).
  > - Decision Trees for non-linear rule capture.
  > - Random Forest Classifier as an ensemble bagging method (~83% accuracy, 0.86 ROC-AUC).
  > - Gradient Boosting Classifier for sequential error correction.
  > Random Forest emerged as the top performer, balancing high precision and recall while minimizing false positives."*

---

## Slide 7: Model Training & Evaluation Results
- **Header**: AI Capstone | Slide 07
- **Title**: Hyperparameter Tuning & Test Performance
- **Speaker Notes**:
  > *"We applied 3-fold cross-validated GridSearchCV across tree estimators and max depth. On the unseen test dataset, our final Random Forest pipeline achieved:
  > - Accuracy: 83.3%
  > - Precision: 78.4%
  > - Recall: 71.2%
  > - F1-Score: 0.746
  > - ROC-AUC: 0.865
  > Feature importance analysis confirmed that Contract Type, Tenure, and Monthly Charges were the strongest predictive drivers."*

---

## Slide 8: Model Serialization & Deployment Architecture
- **Header**: AI Capstone | Slide 08
- **Title**: Serialization & Deployment Architecture
- **Speaker Notes**:
  > *"Rather than retraining the model on every inference request, we serialized the complete trained pipeline into a single file: `model/model.pkl` using Joblib. Our FastAPI service loads this artifact into memory upon startup. The API exposes dedicated endpoints: `/health` for uptime checks, `/predict` for single customer scoring, and `/predict_batch` for high-throughput enterprise batch processing. In addition, the API stratifies customers into Low, Moderate, and High Risk categories."*

---

## Slide 9: API Demonstration & Live Inference
- **Header**: AI Capstone | Slide 09
- **Title**: API Architecture & Live Inference Demonstration
- **Speaker Notes**:
  > *"Here you can see our live FastAPI Swagger interface running on port 8000. When a client application sends a JSON payload with a customer's tenure and plan details, the API responds in under 15 milliseconds with:
  > - The binary prediction (0 or 1),
  > - Human-readable label ('Yes' or 'No'),
  > - The exact probability score (e.g. 76.4%),
  > - And the risk classification: 'High Risk'.
  > This allows CRM systems to immediately trigger retention discounts or customer care calls."*

---

## Slide 10: Conclusion & Business Value
- **Header**: AI Capstone | Slide 10
- **Title**: Summary of Accomplishments & Business Impact
- **Speaker Notes**:
  > *"In conclusion, this Capstone demonstrates the complete lifecycle of production machine learning: from raw tabular data through exploratory analysis, encapsulated preprocessing, model comparison, serialization, and high-performance API serving. By operationalizing this model, a company can target proactive retention interventions, saving significant recurring revenue. Thank you for your time, and I'd be happy to take any questions!"*
