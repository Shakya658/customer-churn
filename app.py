"""
Customer Churn Intelligence Dashboard
======================================
Run:  streamlit run app.py
Requires: models/ directory produced by churn_analysis.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide",
    page_icon="📊",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — subtle professional polish
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    return {
        "rf":         joblib.load("models/random_forest_model.pkl"),
        "lr":         joblib.load("models/logistic_model.pkl"),
        "xgb":        joblib.load("models/xgb_model.pkl"),
        "scaler":     joblib.load("models/scaler.pkl"),
        "features":   joblib.load("models/feature_columns.pkl"),
        "thresholds": joblib.load("models/thresholds.pkl"),
    }

try:
    assets = load_assets()
except FileNotFoundError as e:
    st.error(
        f"⚠️ Model files not found: {e}\n\n"
        "Run all cells in `churn_analysis.ipynb` first to generate the `models/` directory."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — customer profile inputs
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧑‍💼 Customer Profile")
    st.caption("Adjust the inputs to reflect the customer's details.")

    tenure  = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges ($)", 10, 150, 70)

    st.divider()

    contract    = st.selectbox("Contract Type",    ["Month-to-month", "One year", "Two year"])
    internet    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment     = st.selectbox("Payment Method",   [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])

    st.divider()

    senior        = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner       = st.selectbox("Has Partner",     ["Yes", "No"])
    dependents    = st.selectbox("Has Dependents",  ["No", "Yes"])
    phone_service = st.selectbox("Phone Service",   ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup   = st.selectbox("Online Backup",   ["No", "Yes", "No internet service"])
    device_protect  = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support    = st.selectbox("Tech Support",   ["No", "Yes", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV",   ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.divider()

    model_choice = st.selectbox("🤖 Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    predict_btn  = st.button("🔍 Predict Churn", use_container_width=True, type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — must mirror churn_analysis.ipynb exactly
# ──────────────────────────────────────────────────────────────────────────────
def _binary(val: str) -> int:
    """Map Yes/Female/... → 1, No/Male/... → 0 (mirrors LabelEncoder on binary cols)."""
    return 1 if val in ("Yes", "Female", "1") else 0


def prepare_input() -> pd.DataFrame:
    """
    To make sure that the data is loaded properly we need to replicate the notebook's feature engineering pipeline for a single customer.

    Pipeline steps (must match notebook):
      1. Build raw row with all original column names
      2. Binary-encode 2-class columns (same mapping as LabelEncoder in notebook)
      3. One-hot encode multi-class columns (Contract, InternetService, PaymentMethod)
      4. Add tenure_group dummies
      5. Reindex to saved feature_columns (fills any missing dummies with 0)
      6. Cast to float
    """
    total_charges = tenure * monthly  # same fill logic as notebook

    # ── 1. Raw row ──────────────────────────────────────────────────────────
    raw = {
        "tenure":            tenure,
        "MonthlyCharges":    monthly,
        "TotalCharges":      total_charges,
        "SeniorCitizen":     1 if senior == "Yes" else 0,
        "gender":            "Male",            # not collected in sidebar; model-neutral default
        "Partner":           partner,
        "Dependents":        dependents,
        "PhoneService":      phone_service,
        "MultipleLines":     multiple_lines,
        "InternetService":   internet,
        "OnlineSecurity":    online_security,
        "OnlineBackup":      online_backup,
        "DeviceProtection":  device_protect,
        "TechSupport":       tech_support,
        "StreamingTV":       streaming_tv,
        "StreamingMovies":   streaming_movies,
        "Contract":          contract,
        "PaperlessBilling":  paperless,
        "PaymentMethod":     payment,
    }
    df = pd.DataFrame([raw])

    # ── 2. Binary encoding (matches LabelEncoder mapping in notebook) ────────
    #    LabelEncoder sorts labels alphabetically; e.g. No→0, Yes→1.
    binary_map = {
        "gender":           {"Female": 0, "Male": 1},
        "Partner":          {"No": 0, "Yes": 1},
        "Dependents":       {"No": 0, "Yes": 1},
        "PhoneService":     {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        # MultipleLines, OnlineSecurity etc. have 3 values → handled by get_dummies below
    }
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)

    # ── 3. One-hot encode multi-class columns ────────────────────────────────
    df = pd.get_dummies(df, columns=["MultipleLines", "InternetService", "OnlineSecurity",
                                      "OnlineBackup", "DeviceProtection", "TechSupport",
                                      "StreamingTV", "StreamingMovies",
                                      "Contract", "PaymentMethod"])

    # ── 4. Tenure group dummies ──────────────────────────────────────────────
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"],
    )
    df = pd.get_dummies(df, columns=["tenure_group"])

    # ── 5. Align to training feature columns (fills missing dummies with 0) ─
    df = df.reindex(columns=assets["features"], fill_value=0)

    # ── 6. Cast to float ─────────────────────────────────────────────────────
    df = df.astype(float)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────
def get_model():
    mapping = {
        "Random Forest":       (assets["rf"],  False),
        "Logistic Regression": (assets["lr"],  True),
        "XGBoost":             (assets["xgb"], False),
    }
    return mapping[model_choice]


def apply_threshold(prob: float) -> tuple[str, str]:
    """Return (label, css_class) using the tuned per-model threshold."""
    thresh = assets["thresholds"].get(model_choice, 0.5)
    if prob < 0.30:
        return "Low Risk", "low-risk"
    elif prob < thresh:
        return "Medium Risk", "med-risk"
    else:
        return "High Risk", "high-risk"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────
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

# ── Run prediction ────────────────────────────────────────────────────────────
X            = prepare_input()
model, needs_scaling = get_model()
X_model      = assets["scaler"].transform(X) if needs_scaling else X.to_numpy(dtype=float)

prob         = float(model.predict_proba(X_model)[0][1])
risk_label, risk_class = apply_threshold(prob)
thresh       = assets["thresholds"].get(model_choice, 0.5)
prediction   = "Will Churn" if prob >= thresh else "Will Stay"

# ──────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Churn Probability",  f"{prob*100:.1f}%")
k2.metric("Prediction",         prediction)
k3.metric("Risk Segment",       risk_label)
k4.metric("Decision Threshold", f"{thresh*100:.0f}%")

st.progress(prob, text=f"Churn probability: {prob*100:.1f}%")
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Model Explanation", "📊 Model Comparison", "📈 ROC / PR Curves", "📋 Customer Summary"]
)

# ── Tab 1: SHAP ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">SHAP Waterfall — Why this prediction?</p>',
                unsafe_allow_html=True)
    st.caption(
        "SHAP values show how each feature pushed the prediction up (red) or down (blue) "
        "from the model's baseline. The bar width reflects the magnitude of impact."
    )

    # ── Compute SHAP values once; keep a clean 1-D array for the table ─────────
    shap_1d    = None   # shape (n_features,) — class-1 values for this sample
    base_value = None
    explanation = None

    try:
        if model_choice in ("Random Forest", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            raw       = explainer.shap_values(X_model)  # varies by shap version

            # Normalise to a plain 1-D numpy array for class 1:
            #   • Old shap  → list of 2 arrays, each shape (1, n_features)
            #   • New shap  → single array, shape (1, n_features, 2)  OR  (1, n_features)
            if isinstance(raw, list):
                # list[class0, class1] — classic Random Forest output
                shap_1d    = np.array(raw[1]).reshape(-1)          # class-1 row
                ev         = explainer.expected_value
                base_value = float(ev[1] if hasattr(ev, "__len__") else ev)
            elif isinstance(raw, np.ndarray) and raw.ndim == 3:
                # shape (1, n_features, 2) — newer shap with multi-output
                shap_1d    = raw[0, :, 1]
                ev         = explainer.expected_value
                base_value = float(ev[1] if hasattr(ev, "__len__") else ev)
            else:
                # shape (1, n_features) — XGBoost single-output
                shap_1d    = np.array(raw).reshape(-1)
                ev         = explainer.expected_value
                base_value = float(ev[1] if hasattr(ev, "__len__") else ev)

            explanation = shap.Explanation(
                values=shap_1d,
                base_values=base_value,
                data=X.values[0],
                feature_names=X.columns.tolist(),
            )

        else:
            # Logistic Regression
            explainer  = shap.LinearExplainer(model, assets["scaler"].transform(X))
            raw        = explainer.shap_values(X_model)
            shap_1d    = np.array(raw).reshape(-1)
            base_value = float(np.array(explainer.expected_value).reshape(-1)[0])
            explanation = shap.Explanation(
                values=shap_1d,
                base_values=base_value,
                data=X.values[0],
                feature_names=X.columns.tolist(),
            )

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    # ── Waterfall plot ───────────────────────────────────────────────────────
    if explanation is not None:
        try:
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:
            st.warning(f"Waterfall plot error: {e}")

    # ── Top 10 feature contributions table ──────────────────────────────────
    st.markdown('<p class="section-header">Top 10 Feature Contributions</p>',
                unsafe_allow_html=True)

    if shap_1d is not None:
        top_idx = np.argsort(np.abs(shap_1d))[::-1][:10]
        contributions = pd.DataFrame({
            "Feature":     X.columns[top_idx],
            "Value":       X.values[0][top_idx].round(2),
            "SHAP Impact": shap_1d[top_idx].round(4),
        })
        contributions["Direction"] = contributions["SHAP Impact"].apply(
            lambda v: "⬆️ Increases churn" if v > 0 else "⬇️ Reduces churn"
        )
        st.dataframe(contributions, use_container_width=True, hide_index=True)
    else:
        st.info("Feature contributions unavailable — SHAP computation failed above.")

# ── Tab 2: Model Comparison ───────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Churn Probability — All Models</p>',
                unsafe_allow_html=True)

    results = {}
    for name, mdl in [("Random Forest",       assets["rf"]),
                       ("Logistic Regression", assets["lr"]),
                       ("XGBoost",             assets["xgb"])]:
        X_tmp = (assets["scaler"].transform(X)
                 if name == "Logistic Regression"
                 else X.to_numpy(dtype=float))
        results[name] = round(mdl.predict_proba(X_tmp)[0][1] * 100, 2)

    results_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["Churn Probability (%)"]
    )

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(results_df)
    with col_table:
        st.dataframe(results_df.style.format("{:.2f}%"), use_container_width=True)

    # Threshold guide
    st.markdown('<p class="section-header">Model Thresholds</p>', unsafe_allow_html=True)
    thresh_df = pd.DataFrame([
        {"Model": k, "Threshold": f"{v*100:.0f}%",
         "Meaning": f"Predict churn if probability ≥ {v*100:.0f}%"}
        for k, v in assets["thresholds"].items()
    ])
    st.dataframe(thresh_df, use_container_width=True, hide_index=True)

# ── Tab 3: ROC / PR Curves ────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Simulated ROC & Precision-Recall Curves</p>',
                unsafe_allow_html=True)
    st.info(
        "These curves are generated from a **realistic synthetic hold-out set** that "
        "mirrors the Telco dataset's class balance (~26% churn). "
        "For production curves, run the notebook which evaluates on the real test split."
    )

    # Generate a realistic synthetic set (seeded for reproducibility)
    rng = np.random.default_rng(42)
    n   = 1_000
    y_true = rng.choice([0, 1], size=n, p=[0.74, 0.26])

    # Simulate model scores with realistic separation
    def _sim_scores(auc_target: float) -> np.ndarray:
        noise = rng.normal(0, 0.25, n)
        raw   = y_true * auc_target + (1 - y_true) * (1 - auc_target) + noise
        return np.clip(raw, 0, 1)

    model_scores = {
        "Logistic Regression": _sim_scores(0.83),
        "Random Forest":       _sim_scores(0.87),
        "XGBoost":             _sim_scores(0.89),
    }
    palette = {"Logistic Regression": "#3498db",
               "Random Forest":       "#2ecc71",
               "XGBoost":             "#e74c3c"}

    col_roc, col_pr = st.columns(2)

    with col_roc:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, scores in model_scores.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})",
                    color=palette[name], lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with col_pr:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, scores in model_scores.items():
            prec, rec, _ = precision_recall_curve(y_true, scores)
            ax.plot(rec, prec, label=name, color=palette[name], lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

# ── Tab 4: Customer Summary ───────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Customer Profile Summary</p>',
                unsafe_allow_html=True)

    profile = {
        "Tenure":               f"{tenure} months",
        "Monthly Charges":      f"${monthly}",
        "Total Charges (est.)": f"${tenure * monthly:,.0f}",
        "Contract":             contract,
        "Internet Service":     internet,
        "Payment Method":       payment,
        "Tech Support":         tech_support,
        "Paperless Billing":    paperless,
        "Senior Citizen":       senior,
        "Partner":              partner,
        "Dependents":           dependents,
    }
    profile_df = pd.DataFrame(profile.items(), columns=["Attribute", "Value"])
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<p class="section-header">💡 Retention Recommendations</p>',
                unsafe_allow_html=True)

    recs = []
    if prob >= thresh:
        if contract == "Month-to-month":
            recs.append("📄 Offer a discounted **1-year or 2-year contract** to lock in retention.")
        if monthly > 70:
            recs.append("💰 Consider a **loyalty discount** or bundled service offer.")
        if tech_support in ("No", "No internet service"):
            recs.append("🛠️ Proactively offer **Tech Support** — it significantly reduces churn risk.")
        if internet == "Fiber optic":
            recs.append("🌐 Fiber optic customers churn more; check for service quality issues.")
        if payment == "Electronic check":
            recs.append("💳 Encourage switch to **automatic payment** — lower churn in auto-pay customers.")
        if not recs:
            recs.append("⚠️ High churn risk detected. Consider a personalised outreach call.")
    else:
        recs.append("✅ Customer appears stable. Continue standard engagement programme.")

    for rec in recs:
        st.markdown(f"- {rec}")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · Scikit-learn · XGBoost · SHAP  |  "
    "Dataset: Telco Customer Churn (IBM Sample Data)"
)
