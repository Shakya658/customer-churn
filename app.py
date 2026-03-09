import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")
st.write(
    "This tool predicts the probability that a telecom customer will churn based on their service profile."
)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

logistic_model = joblib.load("models/logistic_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------

st.sidebar.header("Customer Information")

model_choice = st.sidebar.selectbox(
    "Prediction Model",
    [
        "Random Forest (Default)",
        "Logistic Regression ",
        "XGBoost"
    ]
)

tenure = st.sidebar.slider(
    "Customer Tenure (months)",
    0, 72, 12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict Churn Risk"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [tenure * monthly_charges],
        "Contract": [contract],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "TechSupport": [tech_support],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method]
    })

    # Encode features
    input_encoded = pd.get_dummies(input_data)

    # Align features
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # ---------------------------------------------------
    # Model Selection
    # ---------------------------------------------------

    if "Logistic" in model_choice:
        model = logistic_model
        input_scaled = scaler.transform(input_encoded)
        prob = model.predict_proba(input_scaled)[0][1]
        model_used = "Logistic Regression"

    elif "Random" in model_choice:
        model = rf_model
        prob = model.predict_proba(input_encoded)[0][1]
        model_used = "Random Forest"

    else:
        model = xgb_model
        prob = model.predict_proba(input_encoded)[0][1]
        model_used = "XGBoost"

    churn_percent = prob * 100

    # ---------------------------------------------------
    # Display Results
    # ---------------------------------------------------

    st.subheader("Prediction Result")

    st.write(f"**Model Used:** {model_used}")

    st.progress(int(churn_percent))

    if prob > 0.5:
        st.error(f"⚠️ High Churn Risk: {churn_percent:.2f}%")
    else:
        st.success(f"✅ Low Churn Risk: {churn_percent:.2f}%")

    st.write("### Probability Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{churn_percent:.2f}%")

    with col2:
        st.metric("Retention Probability", f"{100 - churn_percent:.2f}%")

# ---------------------------------------------------
# About Section
# ---------------------------------------------------

st.markdown("---")

st.markdown(
"""
### About This Project

This machine learning application predicts telecom customer churn using:

• Logistic Regression  
• Random Forest  
• XGBoost  

The models were trained on the **Telco Customer Churn dataset** and evaluated using
classification metrics including **ROC-AUC, precision, recall, and F1-score**.

Default model: **Random Forest** (best at catching the Churn).

Built with **Python, Scikit-Learn, XGBoost, and Streamlit**.
"""
)