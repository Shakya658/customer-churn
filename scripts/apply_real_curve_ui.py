"""Replace the simulated chart section with validated held-out evaluation data."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
README_PATH = ROOT / "README.md"
CURVE_PATH = ROOT / "assets" / "evaluation" / "model_curves.json"


def validate_curve_data() -> dict:
    data = json.loads(CURVE_PATH.read_text(encoding="utf-8"))
    expected = {"Logistic Regression", "Random Forest", "XGBoost"}
    if set(data.get("models", {})) != expected:
        raise ValueError("Evaluation artefact does not contain all three models")
    return data


def update_app() -> None:
    app = APP_PATH.read_text(encoding="utf-8")

    if "import json\n" not in app:
        app = app.replace(
            "import warnings\n",
            "import json\nimport warnings\nfrom pathlib import Path\n",
            1,
        )

    start_marker = "with tab3:\n"
    end_marker = "with tab4:\n"
    start = app.index(start_marker)
    end = app.index(end_marker, start)

    replacement = '''with tab3:
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

'''

    app = app[:start] + replacement + app[end:]
    APP_PATH.write_text(app, encoding="utf-8")


def update_readme(data: dict) -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    if "- Inspect real held-out ROC and precision-recall curves\n" not in readme:
        readme = readme.replace(
            "- Compare model outputs\n",
            "- Compare model outputs\n- Inspect real held-out ROC and precision-recall curves\n",
            1,
        )

    rows = []
    for model_name in ("Logistic Regression", "Random Forest", "XGBoost"):
        metrics = data["models"][model_name]
        rows.append(
            f"| {model_name} | {metrics['roc_auc']:.3f} | "
            f"{metrics['average_precision']:.3f} |"
        )

    section = f'''## Real Held-out Evaluation Curves

The Streamlit ROC and precision-recall charts use predictions from the untouched 20% test split rather than simulated values. The committed models are scored without retraining, and the curve points are stored in `assets/evaluation/model_curves.json`.

The test split contains **{data['test_records']:,} customers**, including **{data['positive_records']:,} churn cases** ({data['positive_rate'] * 100:.1f}% positive rate).

| Model | ROC AUC | Average Precision |
|---|---:|---:|
{chr(10).join(rows)}

The export workflow verifies the 7,043-row source dataset, reconstructs the notebook test split, validates the generated arrays and checks the application syntax before committing the evaluation artefact.

'''

    marker = "## Limitations\n"
    if "## Real Held-out Evaluation Curves\n" in readme:
        section_start = readme.index("## Real Held-out Evaluation Curves\n")
        section_end = readme.index(marker, section_start)
        readme = readme[:section_start] + section + readme[section_end:]
    else:
        readme = readme.replace(marker, section + marker, 1)

    README_PATH.write_text(readme, encoding="utf-8")


def main() -> None:
    data = validate_curve_data()
    update_app()
    update_readme(data)
    print("Updated app.py and README.md with real held-out evaluation curves")


if __name__ == "__main__":
    main()
