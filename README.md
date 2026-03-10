📊 Telco Customer Churn Prediction
Project Overview

Ever wondered why some telecom customers leave their service? This project dives into the Telco Customer Churn dataset to uncover patterns behind customer churn and build machine learning models that predict who’s likely to leave.

It demonstrates a full ML workflow, including:

Cleaning and preprocessing data

Exploratory Data Analysis (EDA)

Feature engineering

Training and evaluating multiple models

Comparing model performance

Deploying a Streamlit web app for interactive predictions

The end result is a user-friendly web app where you can enter customer information and instantly see their churn probability.

📂 Dataset

Source: Telco Customer Churn Dataset

File: WA_Fn-UseC_-Telco-Customer-Churn.csv

Rows: 7,043 customers

Columns: 21 features including tenure, MonthlyCharges, Contract, PaymentMethod, etc.

Target variable: Churn

🧠 Machine Learning Models & Results

We trained three models and tested them on unseen data:

Model	Accuracy	ROC AUC	Notes
Logistic Regression	0.81	0.8519	Solid baseline; high precision for non-churn
Random Forest	0.77	0.8472	Better recall for churn (important for business)
XGBoost	0.79	0.8227	Balanced performance across classes

Default model in the app:
Random Forest is used by default due to its better recall on churn, which is usually more valuable in real-world scenarios. Users can switch models interactively in the app.

🔍 Key Insights

Top factors affecting churn:

Contract Type: Month-to-Month customers are more likely to churn.

Tenure: Shorter tenure → higher churn.

Total Charges: Low or irregular charges indicate disengagement.

Monthly Charges: Higher bills slightly increase churn risk.

Online Security: Customers without it churn more frequently.

📊 Visualizations

The notebook includes helpful visualizations:

Churn distribution

Churn % by categorical features

Numerical feature distributions by churn

Correlation heatmaps

Feature importance plots

ROC & Precision–Recall curves

Confusion matrices

🚀 Streamlit Web App

The project comes with a Streamlit app to make predictions interactive.

Features:

Simple and clean UI

Enter customer information (tenure, monthly charges, contract, etc.)

Choose the prediction model: Logistic Regression, Random Forest (default), XGBoost

Display churn probability and risk level

Example output:
⚠️ High Churn Risk (73.9%)

⚙️ Installation & Setup

Clone the repo:

git clone https://github.com/Shakya658/customer-churn.git
cd customer-churn

Install dependencies:

pip install -r requirements.txt
▶️ Running the App

Start the Streamlit app:

streamlit run app.py

A browser window will open with the interactive app. Enter the customer info, choose a model, and see predictions instantly.

📁 Project Structure
customer-churn/
│
├── Notebooks/
│   └── churn_analysis.ipynb
│
├── models/
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv

(Optional) add /screenshots to show app inputs and outputs directly in README.

🛠 Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn, XGBoost

Streamlit

Joblib



Project Workflow

1️⃣ Business Understanding

The goal of this project is to predict whether a telecom customer is likely to churn so that the company can take preventive actions and improve customer retention.

2️⃣ Data Understanding

The dataset contains customer information such as:

Contract type

Monthly charges

Tenure

Internet service

Payment method

Exploratory analysis was performed to understand feature distributions and relationships with churn.

3️⃣ Data Preparation

Data preprocessing steps included:

Handling missing values

Encoding categorical variables

Feature scaling where necessary

Libraries used:

Pandas

scikit-learn

4️⃣ Model Training

Multiple machine learning models were trained and compared, including:

Logistic Regression

Random Forest

Gradient Boosting

The best-performing model was selected based on evaluation metrics.

5️⃣ Model Evaluation

Models were evaluated using:

Accuracy

Precision

Recall

F1-score

These metrics help assess the model’s ability to correctly identify customers who are likely to churn.

👤 Author

Shirish Shakya
Email: shirishshakya4@gmail.com