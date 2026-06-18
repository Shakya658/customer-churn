# Customer Churn Intelligence Dashboard

An end-to-end machine-learning portfolio project that analyses telecommunications customer churn, compares three classification models, and serves customer-level predictions through an interactive Streamlit application.

The project is designed as a **deployment-ready analytical prototype** rather than a production system. It demonstrates problem framing, data preparation, model evaluation, threshold tuning, explainability, and communication of customer-retention risk.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)
![SHAP](https://img.shields.io/badge/SHAP-0.46-purple)

## Project Overview

Customer churn is costly for subscription businesses because replacing a customer generally requires more effort than retaining an existing one. This project uses the IBM Telco Customer Churn dataset to identify patterns associated with churn and estimate the risk for an individual customer.

The workflow:

1. Cleans and explores 7,043 customer records.
2. Engineers model-ready demographic, service, contract, and billing features.
3. Trains Logistic Regression, Random Forest, and XGBoost classifiers.
4. Tunes each model using cross-validation.
5. Selects decision thresholds using a separate validation set.
6. Explains individual predictions using SHAP.
7. Serves the trained models through a Streamlit dashboard.

## Business Questions

- Which customer characteristics are most associated with churn?
- Which model best separates likely churners from customers who are likely to stay?
- How should the decision threshold change when missed churners are more costly than unnecessary retention outreach?
- Why was a particular customer classified as high risk?
- Which customer risk factors could guide a retention conversation?

## Dataset

**IBM Telco Customer Churn**

| Property | Value |
|---|---:|
| Customer records | 7,043 |
| Original columns | 21 |
| Target | `Churn` (`Yes` / `No`) |
| Non-churn customers | Approximately 73.5% |
| Churn customers | Approximately 26.5% |

The dataset contains customer demographics, subscribed services, contract type, payment method, tenure, monthly charges, total charges, and churn status.

The source CSV is not committed to this repository. Download it from Kaggle and place it in the **repository root** using this exact filename:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Dataset source: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Methodology

### 1. Data preparation

- Converts `TotalCharges` from text to numeric.
- Handles blank `TotalCharges` values for customers with little or no tenure.
- Removes the customer identifier from model inputs.
- Encodes binary categorical variables.
- One-hot encodes multi-category service, contract, and payment variables.
- Creates tenure-group features.

### 2. Data splitting

The data is split using stratification to preserve the churn ratio:

- **70% training set** for model fitting and cross-validation
- **10% validation set** for decision-threshold selection
- **20% test set** for final evaluation

The scaler is fitted only on the training data. The validation set is used for threshold selection, while the test set remains separate from model and threshold tuning.

### 3. Models

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Non-linear ensemble comparison |
| XGBoost | Gradient-boosted model for the strongest predictive performance |

Hyperparameters are selected with `GridSearchCV` using five-fold cross-validation and ROC AUC scoring. Class imbalance is addressed through balanced class weights or `scale_pos_weight`, depending on the model.

### 4. Threshold tuning

A default threshold of `0.50` is not automatically appropriate for churn intervention. Missing a genuine churner can be more costly than contacting a customer who ultimately stays.

The notebook therefore selects model-specific thresholds on the validation set, prioritising recall while still reporting the precision trade-off. The saved thresholds are used directly by the Streamlit application.

### 5. Explainability

The application uses SHAP to show how individual customer features push a prediction towards higher or lower churn risk. This makes the output more useful than a probability alone and supports customer-level investigation.

## Evaluation

Models are compared using:

- ROC AUC
- Precision
- Recall
- F1 score
- Confusion matrices
- Precision-recall curves

In the current notebook workflow, XGBoost provides the strongest overall ROC AUC. Exact values are retained in the executed notebook outputs because they can change when the data split, random seed, dependency versions, or tuning grid changes.

This avoids presenting a model metric without its corresponding experimental context. The notebook is the source of truth for the latest evaluation run.

## Key Analytical Findings

The analysis consistently identifies the following as important churn indicators:

- Short customer tenure
- Month-to-month contracts
- Higher monthly charges
- Electronic-check payment
- Fibre-optic internet service
- Lack of online security or technical-support services

These relationships are observational rather than causal. They identify useful retention segments but do not prove that changing one feature will prevent churn.

## Streamlit Application

The dashboard allows a user to:

- Enter a customer profile
- Select Logistic Regression, Random Forest, or XGBoost
- View the predicted churn probability
- See the model-specific decision threshold
- Assign the customer to a low-, medium-, or high-risk segment
- Inspect the strongest feature contributions through SHAP
- Compare outputs across the three models

Run the application with:

```bash
streamlit run app.py
```

## Repository Structure

```text
customer-churn/
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── notebooks/
│   └── churn_analysis.ipynb
├── models/
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── thresholds.pkl
└── assets/
    ├── plots/
    └── screenshots/
```

The source dataset must be downloaded separately and placed in the repository root.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- The IBM Telco Customer Churn CSV

### 1. Clone the repository

```bash
git clone https://github.com/Shakya658/customer-churn.git
cd customer-churn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install notebook seaborn
```

### 4. Add the dataset

Place the downloaded CSV here:

```text
customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5. Run the analysis notebook

Launch Jupyter from the repository root:

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

Run all cells to reproduce the analysis and regenerate the serialized model artefacts.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open the local address displayed by Streamlit, normally:

```text
http://localhost:8501
```

## Training and Inference Contract

The application must reproduce the feature-engineering steps used during training. The saved `feature_columns.pkl` file defines the required feature order.

At inference time, `prepare_input()`:

1. Builds a one-row customer record.
2. Applies the training-time binary mappings.
3. One-hot encodes multi-category variables.
4. Creates tenure-group indicators.
5. Reindexes the row against the saved feature list.
6. Fills absent dummy variables with zero.
7. Applies scaling only to Logistic Regression.

This alignment prevents missing-column and feature-order errors when a single customer profile does not contain every possible category.

## Important Limitations

- The project uses a public sample dataset rather than live company data.
- The dataset does not include customer-interaction history, competitor offers, service outages, or retention-campaign outcomes.
- Predicted risk should support investigation, not automatically determine customer treatment.
- SHAP explanations describe model behaviour and should not be interpreted as causal proof.
- The application is a portfolio prototype and does not include production monitoring, authentication, drift detection, or automated retraining.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Author

**Shirish Man Shakya**  
Data Analyst | Business Intelligence | Predictive Analytics

- [Portfolio](https://shakya658.github.io/portfolio/)
- [LinkedIn](https://linkedin.com/in/shirish-man-shakya)
- [GitHub](https://github.com/Shakya658)
