Customer Churn Analysis

Project Overview

This project analyzes the Telco Customer Churn dataset to understand factors influencing customer churn and predict churn using machine learning models. The notebook demonstrates a complete workflow from data cleaning and exploratory analysis to feature engineering and predictive modeling.

The goal is to:

Identify patterns leading to customer churn

Understand which features most affect churn

Train and evaluate Logistic Regression, Random Forest, and XGBoost models

Prepare models for deployment as a web app or API

Dataset

Source: Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)

Rows: 7043

Columns: 21 features (e.g., tenure, MonthlyCharges, Contract, PaymentMethod, Churn)

Project Structure
.
├── Data/                       # Dataset CSV
├── Notebooks/
│   └── churn_analysis.ipynb    # Main analysis notebook
├── Models/
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── README.md                   # This file
Requirements

Python 3.x

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

Install packages:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost
How to Run

Open Notebooks/churn_analysis.ipynb in Jupyter Notebook or JupyterLab.

Run cells sequentially to reproduce the analysis, visualizations, and model evaluations.

Key Features

Data Cleaning:

Converted TotalCharges to numeric and handled missing values intelligently.

Exploratory Data Analysis:

Churn percentages by category for easy interpretation

Distribution plots and boxplots for numerical features

Correlation heatmaps and pairplots to understand feature relationships

Feature Engineering:

Label encoding for binary features

One-hot encoding for multi-class features

Tenure groups created and encoded

Models & Evaluation:

Logistic Regression, Random Forest, and XGBoost

Evaluation metrics include classification reports, ROC AUC, Precision-Recall, and confusion matrices

Feature Importance:

Top 5 features highlighted with plain-language explanations

Models & Performance Highlights
Model	Accuracy	ROC AUC	Notes
Logistic Regression	~81%	0.85	Baseline model
Random Forest	~82%	0.87	Better recall for churn class
XGBoost	~82%	0.87	Additional model with strong predictive power

Top 5 Features Affecting Churn:

Contract_Month-to-month: Customers on month-to-month contracts churn more.

tenure: Shorter tenure → higher churn.

TotalCharges: Lower or irregular total charges sometimes indicate disengaged customers.

MonthlyCharges: Higher monthly bills slightly increase churn.

OnlineSecurity_No: Lack of online security service correlates with higher churn.

Visualizations

Churn distribution

Churn % by categorical features

Numerical feature distributions by churn

Feature importance barplots

ROC and Precision-Recall curves for model comparison

Confusion matrix heatmaps

Deployment

Trained models (logistic_model.pkl, random_forest_model.pkl) and scaler (scaler.pkl) are saved and ready for use in a web app or API for predicting customer churn.

Author

Shirish Shakya | shirishshakya4@gmail.com
