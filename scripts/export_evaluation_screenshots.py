"""Generate README-ready screenshots from validated held-out evaluation data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CURVE_PATH = ROOT / "assets" / "evaluation" / "model_curves.json"
SCREENSHOT_DIR = ROOT / "assets" / "screenshots"


def main() -> None:
    data = json.loads(CURVE_PATH.read_text(encoding="utf-8"))
    expected = {"Logistic Regression", "Random Forest", "XGBoost"}
    if set(data.get("models", {})) != expected:
        raise ValueError("Evaluation data does not contain all three models")

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 6))
    for name, metrics in data["models"].items():
        axis.plot(
            metrics["false_positive_rate"],
            metrics["true_positive_rate"],
            label=f"{name} (AUC={metrics['roc_auc']:.3f})",
            linewidth=2,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Real Held-out ROC Curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(SCREENSHOT_DIR / "real_roc_curves.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    for name, metrics in data["models"].items():
        axis.plot(
            metrics["recall"],
            metrics["precision"],
            label=f"{name} (AP={metrics['average_precision']:.3f})",
            linewidth=2,
        )
    axis.axhline(
        data["positive_rate"],
        linestyle="--",
        linewidth=1,
        label=f"Class prevalence ({data['positive_rate']:.3f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Real Held-out Precision-Recall Curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(SCREENSHOT_DIR / "real_precision_recall_curves.png", dpi=180)
    plt.close(figure)

    print("Generated real evaluation screenshots")


if __name__ == "__main__":
    main()
