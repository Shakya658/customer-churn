📊 Telco Customer Churn Prediction
Project Overview

This project analyzes the Telco Customer Churn dataset to understand the key factors influencing customer churn and to build machine learning models capable of predicting whether a customer is likely to leave a telecom service.

The project demonstrates a complete machine learning workflow, including:

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Model training and evaluation

Model comparison

Deployment using a Streamlit web application

The final result is an interactive web app that predicts customer churn probability based on customer information.

📂 Dataset

Source: Telco Customer Churn Dataset
File: WA_Fn-UseC_-Telco-Customer-Churn.csv

Dataset characteristics:

Rows: 7043 customers

Features: 21 columns

Target variable: Churn

Example features:

tenure

MonthlyCharges

TotalCharges

Contract

InternetService

OnlineSecurity

TechSupport

PaymentMethod

PaperlessBilling

🧠 Machine Learning Models & Test Results

Three models were trained and evaluated on the test set:

Model	Accuracy	ROC AUC	Notes
Logistic Regression	0.81	0.8519	Strong baseline with good precision for non-churn
Random Forest	0.77	0.8472	Better recall for churn class; slightly lower accuracy
XGBoost	0.79	0.8227	Moderate performance; balanced between classes

Default Model for the App:
Random Forest is used as default for predictions due to its better recall on the churn class, which is often more important in business contexts. Users can switch models interactively in the app.

🔍 Key Insights from Analysis

Top factors influencing churn:

1️⃣ Contract Type

Customers on Month-to-Month contracts churn more frequently.

2️⃣ Tenure

Customers with shorter tenure are more likely to churn.

3️⃣ Total Charges

Lower or inconsistent total charges can indicate disengaged customers.

4️⃣ Monthly Charges

Higher monthly costs slightly increase churn probability.

5️⃣ Online Security

Customers without online security services churn more often.

📊 Visualizations

The notebook includes several visualizations to aid understanding:

Churn distribution

Churn percentages by categorical features

Numerical feature distributions by churn

Correlation heatmaps

Feature importance plots

ROC curves

Precision–Recall curves

Confusion matrices

🚀 Streamlit Web App

The project includes a Streamlit web application:

App Features:

User-friendly UI

Input customer information

Select the prediction model:

Logistic Regression

Random Forest (default)

XGBoost

Predict churn probability and display risk level

Example prediction output:

⚠️ High Churn Risk (73.9%)
⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

Install required packages:

pip install -r requirements.txt
▶️ Running the Web App

Run the Streamlit application:

streamlit run app.py

Then open the browser link shown in the terminal.

📁 Project Structure
customer-churn-prediction
│
├── Notebooks
│   └── churn_analysis.ipynb
│
├── models
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
├── README.md
│
└── data
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
🛠 Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

Streamlit

Joblib

👤 Author

Shirish Shakya
Email: shirishshakya4@gmail.com