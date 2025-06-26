Customer Churn Analysis
Project Overview
This project analyzes the Telco Customer Churn dataset to understand factors influencing customer churn and to predict churn using machine learning models.

Project Structure
Data/ : Contains the dataset CSV file

Notebooks/ : Contains the Jupyter notebook churn_analysis.ipynb

README.md : This file, describing the project

How to Run
Open the notebook notebooks/churn_analysis.ipynb in Jupyter Notebook or JupyterLab

Run each cell sequentially to reproduce the entire analysis

Requirements
Python 3.x

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

Install the required packages with:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Summary
Loaded and cleaned the dataset, handling missing and categorical data

Performed exploratory data analysis with visualizations to uncover churn patterns

Built and evaluated three models: Logistic Regression, Random Forest, and XGBoost

Compared model performances using classification reports, ROC AUC scores, and confusion matrices

Saved trained models for reuse

Models & Performance Highlights
Model	Accuracy	ROC AUC	Notes
Logistic Regression	~79%	~0.83	Baseline model
Random Forest	~76%	~0.82	Better recall for churn class
XGBoost	~76%	~0.83	Additional model, needs tuning

Next Steps
Hyperparameter tuning for improved model performance

Model interpretability using SHAP or feature importance analysis

Deployment as a web app or API

Author
Shirish Shakya | shirishshakya4@gmail.com

