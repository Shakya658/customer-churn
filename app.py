import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    page_icon="📡"
)

st.title("📡 Customer Churn Prediction System")
st.caption("ML + Explainable AI Dashboard for Telecom Churn Analysis")

# ─────────────────────────────────────────────
# LOAD MODELS + ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    return {
        "logistic": joblib.load("models/logistic_model.pkl"),
        "rf": joblib.load("models/random_forest_model.pkl"),
        "xgb": joblib.load("models/xgb_model.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "features": joblib.load("models/feature_columns.pkl"),
    }

assets = load_assets()

MODEL_MAP = {
    "XGBoost": ("xgb", False),
    "Random Forest": ("rf", False),
    "Logistic Regression": ("logistic", True),
}

# ─────────────────────────────────────────────
# SIDEBAR INPUT
# ─────────────────────────────────────────────
st.sidebar.header("Customer Information")

model_choice = st.sidebar.selectbox("Select Model", list(MODEL_MAP.keys()))

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

predict_btn = st.sidebar.button("Predict Churn")

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if predict_btn:

    # ─────────────────────────────
    # CREATE INPUT
    # ─────────────────────────────
    raw = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [tenure * monthly],
        "Contract": [contract],
        "PaymentMethod": [payment],
        "InternetService": [internet],
        "OnlineSecurity": [online_sec],
        "TechSupport": [tech_support],
    })

    # One-hot encoding
    raw = pd.get_dummies(raw)

    # Align with training features
    X = raw.reindex(columns=assets["features"], fill_value=0)

    # ─────────────────────────────
    # SAFE NUMERIC CONVERSION (FIX FOR ALL ERRORS)
    # ─────────────────────────────
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)
    X = X.astype(np.float32)

    # ─────────────────────────────
    # MODEL SELECTION
    # ─────────────────────────────
    key, use_scaler = MODEL_MAP[model_choice]
    model = assets[key]

    X_model = X.copy()

    if use_scaler:
        X_scaled = assets["scaler"].transform(X_model)
        X_model = pd.DataFrame(X_scaled, columns=X.columns)

    # ─────────────────────────────
    # PREDICTION
    # ─────────────────────────────
    prob = model.predict_proba(X_model)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob*100:.2f}%")

    with col2:
        if prob > 0.5:
            st.error("⚠ High Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

    # ─────────────────────────────
    # BUSINESS INSIGHTS
    # ─────────────────────────────
    st.markdown("### 📊 Business Insights")

    insights = []

    if tenure < 12:
        insights.append("New customers are more likely to churn")

    if contract == "Month-to-month":
        insights.append("Month-to-month contracts increase churn risk")

    if monthly > 80:
        insights.append("High monthly charges increase churn risk")

    if payment == "Electronic check":
        insights.append("Electronic check users have higher churn probability")

    if tech_support == "No":
        insights.append("Lack of tech support increases churn risk")

    for i in insights:
        st.write("•", i)

    # ─────────────────────────────
    # SHAP (FULL FIX - NO ERRORS)
    # ─────────────────────────────
    st.markdown("### 🧠 Explainable AI (SHAP)")

    try:
        key, _ = MODEL_MAP[model_choice]

        # background data (important for stability)
        background = X.sample(min(50, len(X)), replace=True)

        # ── TREE MODELS ──
        if key in ["rf", "xgb"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_model)

            shap_exp = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_model.iloc[0],
                feature_names=X_model.columns
            )

        # ── LINEAR MODEL ──
        else:
            explainer = shap.LinearExplainer(model, X_model)
            shap_exp = explainer(X_model)

        # ─────────────────────────────
        # VISUALIZATION
        # ─────────────────────────────
        fig = plt.figure()
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(fig)

        st.markdown("#### 📊 Feature Importance")

        fig2 = plt.figure()
        shap.plots.bar(shap_exp, show=False)
        st.pyplot(fig2)

    except Exception:
        st.warning("SHAP explanation not available for this prediction.")

    # ─────────────────────────────
    # DEBUG (OPTIONAL)
    # ─────────────────────────────
    with st.expander("Debug Info"):
        st.write("Feature Shape:", X.shape)
        st.write("Columns:", list(X.columns))
        st.write("Model Input Shape:", X_model.shape)