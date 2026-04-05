# 📊 Customer Churn Intelligence Dashboard

> An end-to-end machine learning project — from exploratory data analysis to a production-ready Streamlit dashboard — predicting customer churn for a telecommunications provider.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)
![SHAP](https://img.shields.io/badge/SHAP-0.46-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🗂 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Pipeline Contract](#pipeline-contract)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

Customer churn — when a customer stops using a service — is one of the most costly problems in subscription businesses. This project uses the **IBM Telco Customer Churn** dataset (7,043 customers, 21 features) to:

1. **Explore** patterns driving churn through EDA
2. **Engineer** features that improve model signal
3. **Train and compare** three classifiers: Logistic Regression, Random Forest, and XGBoost
4. **Explain** individual predictions using SHAP values
5. **Serve** everything through an interactive Streamlit dashboard with actionable retention recommendations

---

## Demo

```
streamlit run app.py
```

The dashboard lets you input any customer profile and instantly see:

- **Churn probability** with a tuned decision threshold
- **SHAP waterfall** explaining which features drove the prediction
- **Top 10 feature contributions** table
- **Model comparison** across all three classifiers
- **Retention recommendations** tailored to the customer's risk factors

---
##Prjoect structure

customer-churn/
│
├── .devcontainer/              
├── .github/                    
│
├── data/                       
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/                     
│
├── notebooks/                   (lowercase)
│   └── churn_analysis.ipynb
│
├── assets/                     
│   ├── plots/                  
│   └── screenshots/            
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md

## Dataset

**IBM Telco Customer Churn** — publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Features | 21 (demographics, services, billing) |
| Target | `Churn` — Yes / No |
| Class balance | ~73.5% No · 26.5% Yes |

Download the CSV, rename it `WA_Fn-UseC_-Telco-Customer-Churn.csv`, and place it in the project root before running the notebook.

---

## Methodology

### 1. Data Cleaning
- `TotalCharges` stored as `object` due to blank strings for new customers (tenure = 0)
- Coerced to numeric; 11 NaN rows filled with `MonthlyCharges × tenure` to preserve data

### 2. Feature Engineering
| Step | Detail |
|---|---|
| Binary encoding | `LabelEncoder` on 2-class columns (Partner, Dependents, PhoneService, etc.) |
| One-hot encoding | `pd.get_dummies` on Contract, InternetService, PaymentMethod, and service add-ons |
| Tenure bins | Ordinal `tenure_group` feature (0–12m, 13–24m, 25–48m, 49–60m, 61–72m) |
| Bool → int cast | All boolean dummy columns cast to `int` for XGBoost compatibility |

### 3. Train / Validation / Test Split
- **70% train · 10% validation · 20% test** — stratified on the target
- Scaler fitted **only on the training set** to prevent data leakage

### 4. Hyperparameter Tuning
All three models tuned with `GridSearchCV` (5-fold CV, `roc_auc` scoring):

- **Logistic Regression** — C, penalty (L1/L2), solver
- **Random Forest** — n_estimators, max_depth, min_samples_leaf, max_features
- **XGBoost** — max_depth, learning_rate, n_estimators, subsample, colsample_bytree; `scale_pos_weight` set to the negative/positive class ratio to handle imbalance

### 5. Threshold Tuning
Default 0.5 thresholds optimise accuracy but underserve recall in imbalanced churn scenarios. The XGBoost threshold is tuned on the **validation set** to achieve ≥ 80% recall using the precision-recall curve.

---

## Models & Results

| Model | Test AUC | Notes |
|---|---|---|
| Logistic Regression | ~0.84 | Strong baseline; interpretable coefficients |
| Random Forest | ~0.87 | Robust to outliers; no scaling needed |
| **XGBoost** | **~0.89** | Best overall; handles class imbalance via `scale_pos_weight` |

*Exact values depend on your random seed and GridSearchCV run.*

**Top churn predictors (Random Forest feature importance):**
1. `tenure` — longer tenure = much lower churn risk
2. `Contract_Month-to-month` — highest churn rate of any contract type
3. `MonthlyCharges` — higher charges correlate with churn
4. `InternetService_Fiber optic` — Fiber customers churn more than DSL
5. `TotalCharges` — proxy for customer lifetime value

---

## Getting Started

### Prerequisites
- Python 3.10+
- The dataset CSV (see [Dataset](#dataset))

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Shakya658/customer-churn.git
cd customer-churn-intelligence

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset in the project root
#    WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Generate Models

Open and run all cells in `churn_analysis.ipynb`. This will:
- Train and tune all three models (~5–10 minutes depending on hardware)
- Save serialised artefacts to the `models/` directory
- Generate EDA plots as `.png` files

```bash
jupyter notebook churn_analysis.ipynb
```

---

## Running the App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

> **Note:** The `models/` directory must exist before launching the app. Run the notebook first.

---

## Pipeline Contract

The app's `prepare_input()` function **must replicate the notebook's feature engineering exactly** — the models only accept a fixed column schema saved in `models/feature_columns.pkl`.

| Step | Notebook | app.py |
|---|---|---|
| Binary cols | `LabelEncoder` on 2-unique cols | Manual dict mapping (same alphabetical order) |
| Multi-cat cols | `pd.get_dummies` on Contract, Internet, Payment + service cols | Same `pd.get_dummies` call |
| Tenure bins | `pd.cut` → `pd.get_dummies` | Same bins and labels |
| Column alignment | Source of truth | `reindex(columns=feature_columns, fill_value=0)` |
| Scaling | `StandardScaler` fit on `X_train` | `scaler.transform(X)` for Logistic Regression only |
| XGBoost input | `X.to_numpy(dtype=float)` | `X.to_numpy(dtype=float)` |

The `reindex` call in `prepare_input()` is the safety net — even if a user's input doesn't trigger every dummy column, the array fed to the model always has the correct shape and column order.

---

## Key Design Decisions

**Why numpy arrays for XGBoost?**
XGBoost raises feature-name mismatch warnings (and errors on some versions) when pandas DataFrames with column names are passed at inference time if the model was trained on arrays. Training on `X.to_numpy()` and inferring on the same eliminates this entirely.

**Why separate validation and test sets?**
The validation set is used exclusively for threshold tuning. Using the test set for this would constitute data leakage — the test set is touched only once, for final reported metrics.

**Why `class_weight='balanced'` and `scale_pos_weight`?**
The dataset is ~27% positive (churn). Without correction, models optimise accuracy by predicting "No churn" for most customers — useless in practice. Balanced weights penalise missed churners more heavily.

**Why tune for recall ≥ 80%?**
In churn prediction the cost of a **false negative** (missing a churner) typically exceeds the cost of a **false positive** (unnecessary retention outreach). A recall-optimised threshold captures more at-risk customers at the expense of some precision.

---

## License

MIT — free to use, modify, and distribute with attribution.
