"""
Customer Churn Intelligence Dashboard
======================================
Run:  streamlit run app.py
Requires: models/ directory produced by notebooks/churn_analysis.ipynb
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide",
    page_icon="📊",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #e74c3c;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .low-risk  { border-left-color: #2ecc71 !important; }
    .med-risk  { border-left-color: #f39c12 !important; }
    .high-risk { border-left-color: #e74c3c !important; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    return {
        "rf": joblib.load("models/random_forest_model.pkl"),
        "lr": joblib.load("models/logistic_model.pkl"),
        "xgb": joblib.load("models/xgb_model.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "features": joblib.load("models/feature_columns.pkl"),
        "thresholds": joblib.load("models/thresholds.pkl"),
    }


try:
    assets = load_assets()
except FileNotFoundError as error:
    st.error(
        f"⚠️ Model files not found: {error}\n\n"
        "Run all cells in `notebooks/churn_analysis.ipynb` first to generate the `models/` directory."
    )
    st.stop()


with st.sidebar:
    st.header("🧑‍💼 Customer Profile")
    st.caption("Adjust the inputs to reflect the customer's details.")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges ($)", 10, 150, 70)

    st.divider()

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    st.divider()

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.divider()

    model_choice = st.selectbox("🤖 Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    predict_btn = st.button("🔍 Predict Churn", use_container_width=True, type="primary")


def prepare_input() -> pd.DataFrame:
    """Replicate the notebook feature-engineering steps for one customer."""
    total_charges = tenure * monthly

    raw = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }
    df = pd.DataFrame([raw])

    binary_map = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for column, mapping in binary_map.items():
        df[column] = df[column].map(mapping)

    df = pd.get_dummies(
        df,
        columns=[
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ],
    )

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"],
    )
    df = pd.get_dummies(df, columns=["tenure_group"])
    df = df.reindex(columns=assets["features"], fill_value=0)
    return df.astype(float)


def get_model():
    mapping = {
        "Random Forest": (assets["rf"], False),
        "Logistic Regression": (assets["lr"], True),
        "XGBoost": (assets["xgb"], False),
    }
    return mapping[model_choice]


def apply_threshold(probability: float) -> tuple[str, str]:
    threshold = assets["thresholds"].get(model_choice, 0.5)
    if probability < 0.30:
        return "Low Risk", "low-risk"
    if probability < threshold:
        return "Medium Risk", "med-risk"
    return "High Risk", "high-risk"


st.title("📊 Customer Churn Intelligence Dashboard")
st.caption(
    "Enter a customer profile in the sidebar and click **Predict Churn** "
    "to get a real-time churn probability, model explanation, and actionable insights."
)

if not predict_btn:
    col1, col2, col3 = st.columns(3)
    col1.info("👈 Fill in the customer profile in the sidebar")
    col2.info("🤖 Choose a model and click **Predict Churn**")
    col3.info("📈 View probability, SHAP explanation, and model comparison")
    st.stop()

X = prepare_input()
model, needs_scaling = get_model()
X_model = assets["scaler"].transform(X) if needs_scaling else X.to_numpy(dtype=float)

prob = float(model.predict_proba(X_model)[0][1])
risk_label, risk_class = apply_threshold(prob)
thresh = assets["thresholds"].get(model_choice, 0.5)
prediction = "Will Churn" if prob >= thresh else "Will Stay"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Churn Probability", f"{prob * 100:.1f}%")
k2.metric("Prediction", prediction)
k3.metric("Risk Segment", risk_label)
k4.metric("Decision Threshold", f"{thresh * 100:.0f}%")

st.progress(prob, text=f"Churn probability: {prob * 100:.1f}%")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Model Explanation", "📊 Model Comparison", "📈 ROC / PR Curves", "📋 Customer Summary"]
)

with tab1:
    st.markdown(
        '<p class="section-header">SHAP Waterfall — Why this prediction?</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "SHAP values show how each feature pushed the prediction up or down from the model baseline."
    )

    shap_1d = None
    base_value = None
    explanation = None

    try:
        if model_choice in ("Random Forest", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(X_model)

            if isinstance(raw, list):
                shap_1d = np.array(raw[1]).reshape(-1)
                expected = explainer.expected_value
                base_value = float(expected[1] if hasattr(expected, "__len__") else expected)
            elif isinstance(raw, np.ndarray) and raw.ndim == 3:
                shap_1d = raw[0, :, 1]
                expected = explainer.expected_value
                base_value = float(expected[1] if hasattr(expected, "__len__") else expected)
            else:
                shap_1d = np.array(raw).reshape(-1)
                expected = explainer.expected_value
                base_value = float(expected[1] if hasattr(expected, "__len__") else expected)
        else:
            explainer = shap.LinearExplainer(model, assets["scaler"].transform(X))
            raw = explainer.shap_values(X_model)
            shap_1d = np.array(raw).reshape(-1)
            base_value = float(np.array(explainer.expected_value).reshape(-1)[0])

        explanation = shap.Explanation(
            values=shap_1d,
            base_values=base_value,
            data=X.values[0],
            feature_names=X.columns.tolist(),
        )
    except Exception as error:
        st.warning(f"SHAP explanation unavailable: {error}")

    if explanation is not None:
        try:
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as error:
            st.warning(f"Waterfall plot error: {error}")

    st.markdown(
        '<p class="section-header">Top 10 Feature Contributions</p>',
        unsafe_allow_html=True,
    )
    if shap_1d is not None:
        top_idx = np.argsort(np.abs(shap_1d))[::-1][:10]
        contributions = pd.DataFrame(
            {
                "Feature": X.columns[top_idx],
                "Value": X.values[0][top_idx].round(2),
                "SHAP Impact": shap_1d[top_idx].round(4),
            }
        )
        contributions["Direction"] = contributions["SHAP Impact"].apply(
            lambda value: "⬆️ Increases churn" if value > 0 else "⬇️ Reduces churn"
        )
        st.dataframe(contributions, use_container_width=True, hide_index=True)
    else:
        st.info("Feature contributions unavailable — SHAP computation failed above.")

with tab2:
    st.markdown(
        '<p class="section-header">Churn Probability — All Models</p>',
        unsafe_allow_html=True,
    )
    results = {}
    for name, candidate_model in [
        ("Random Forest", assets["rf"]),
        ("Logistic Regression", assets["lr"]),
        ("XGBoost", assets["xgb"]),
    ]:
        candidate_input = (
            assets["scaler"].transform(X)
            if name == "Logistic Regression"
            else X.to_numpy(dtype=float)
        )
        results[name] = round(candidate_model.predict_proba(candidate_input)[0][1] * 100, 2)

    results_df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["Churn Probability (%)"],
    )
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(results_df)
    with col_table:
        st.dataframe(results_df.style.format("{:.2f}%"), use_container_width=True)

    st.markdown('<p class="section-header">Model Thresholds</p>', unsafe_allow_html=True)
    threshold_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Threshold": f"{value * 100:.0f}%",
                "Meaning": f"Predict churn if probability ≥ {value * 100:.0f}%",
            }
            for name, value in assets["thresholds"].items()
        ]
    )
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown(
        '<p class="section-header">Held-out ROC & Precision-Recall Curves</p>',
        unsafe_allow_html=True,
    )

    curve_path = Path("assets/evaluation/model_curves.json")
    try:
        curve_data = json.loads(curve_path.read_text(encoding="utf-8"))
        expected_models = {"Logistic Regression", "Random Forest", "XGBoost"}
        if set(curve_data.get("models", {})) != expected_models:
            raise ValueError("Evaluation data does not contain all three models")

        st.caption(
            f"Real held-out test evaluation: {curve_data['test_records']:,} customers, "
            f"{curve_data['positive_records']:,} churn cases "
            f"({curve_data['positive_rate'] * 100:.1f}% positive rate)."
        )

        palette = {
            "Logistic Regression": "#3498db",
            "Random Forest": "#2ecc71",
            "XGBoost": "#e74c3c",
        }

        col_roc, col_pr = st.columns(2)
        with col_roc:
            figure, axis = plt.subplots(figsize=(5, 4))
            for name, metrics in curve_data["models"].items():
                axis.plot(
                    metrics["false_positive_rate"],
                    metrics["true_positive_rate"],
                    label=f"{name} (AUC={metrics['roc_auc']:.3f})",
                    color=palette[name],
                    lw=2,
                )
            axis.plot([0, 1], [0, 1], "k--", lw=1)
            axis.set_xlabel("False Positive Rate")
            axis.set_ylabel("True Positive Rate")
            axis.set_title("ROC Curves", fontweight="bold")
            axis.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(figure, clear_figure=True)

        with col_pr:
            figure, axis = plt.subplots(figsize=(5, 4))
            for name, metrics in curve_data["models"].items():
                axis.plot(
                    metrics["recall"],
                    metrics["precision"],
                    label=f"{name} (AP={metrics['average_precision']:.3f})",
                    color=palette[name],
                    lw=2,
                )
            axis.axhline(
                curve_data["positive_rate"],
                linestyle="--",
                linewidth=1,
                label="Class prevalence",
            )
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.set_title("Precision-Recall Curves", fontweight="bold")
            axis.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(figure, clear_figure=True)

    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        st.warning(
            "Held-out evaluation curves are unavailable. "
            f"Run `python scripts/export_evaluation_curves.py` to regenerate them. Details: {error}"
        )

with tab4:
    st.markdown(
        '<p class="section-header">Customer Profile Summary</p>',
        unsafe_allow_html=True,
    )
    profile = {
        "Gender": gender,
        "Tenure": f"{tenure} months",
        "Monthly Charges": f"${monthly}",
        "Total Charges (est.)": f"${tenure * monthly:,.0f}",
        "Contract": contract,
        "Internet Service": internet,
        "Payment Method": payment,
        "Tech Support": tech_support,
        "Paperless Billing": paperless,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
    }
    profile_df = pd.DataFrame(profile.items(), columns=["Attribute", "Value"])
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        '<p class="section-header">💡 Retention Recommendations</p>',
        unsafe_allow_html=True,
    )

    recommendations = []
    if prob >= thresh:
        if contract == "Month-to-month":
            recommendations.append("📄 Offer a discounted **1-year or 2-year contract** to support retention.")
        if monthly > 70:
            recommendations.append("💰 Consider a **loyalty discount** or bundled service offer.")
        if tech_support in ("No", "No internet service"):
            recommendations.append("🛠️ Proactively offer **Tech Support** as part of a retention conversation.")
        if internet == "Fiber optic":
            recommendations.append("🌐 Review service quality and support history for this fibre customer.")
        if payment == "Electronic check":
            recommendations.append("💳 Consider offering an automatic-payment option.")
        if not recommendations:
            recommendations.append("⚠️ High churn risk detected. Consider personalised outreach.")
    else:
        recommendations.append("✅ Customer appears stable. Continue standard engagement.")

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

st.divider()
st.caption(
    "Built with Streamlit · Scikit-learn · XGBoost · SHAP | "
    "Dataset: Telco Customer Churn (IBM Sample Data)"
)
