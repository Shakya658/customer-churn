"""Export real ROC and precision-recall curves from the notebook test split.

The script executes only the notebook cells needed to reconstruct the final encoded
X_test and y_test. It then scores that held-out split with the model artefacts
already committed to the repository. It does not retrain or overwrite the models.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "churn_analysis.ipynb"
OUTPUT_PATH = ROOT / "assets" / "evaluation" / "model_curves.json"

MODEL_PATHS = {
    "Logistic Regression": ROOT / "models" / "logistic_model.pkl",
    "Random Forest": ROOT / "models" / "random_forest_model.pkl",
    "XGBoost": ROOT / "models" / "xgb_model.pkl",
}


def _clean_notebook_source(source: str) -> str:
    """Remove notebook-only shell and magic commands before execution."""
    cleaned: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            continue
        if "google.colab" in stripped or "drive.mount(" in stripped:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _matches_feature_contract(candidate: Any, feature_columns: list[str]) -> bool:
    if not isinstance(candidate, pd.DataFrame):
        return False
    return candidate.shape[1] == len(feature_columns) and list(candidate.columns) == feature_columns


def reconstruct_test_split(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
    }

    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue

            source = _clean_notebook_source(cell.source)
            if not source.strip():
                continue

            try:
                exec(compile(source, f"{NOTEBOOK_PATH.name}:cell_{index}", "exec"), namespace)
            except Exception as error:
                # Cells after the data split may contain plotting, tuning or environment-
                # specific operations. We only tolerate an error after the exact split
                # needed by the deployed model contract has already been reconstructed.
                x_test = namespace.get("X_test")
                y_test = namespace.get("y_test")
                if _matches_feature_contract(x_test, feature_columns) and y_test is not None:
                    break
                raise RuntimeError(f"Notebook cell {index} failed before test split was ready") from error

            x_test = namespace.get("X_test")
            y_test = namespace.get("y_test")
            if _matches_feature_contract(x_test, feature_columns) and y_test is not None:
                y_series = pd.Series(y_test).reset_index(drop=True)
                x_frame = x_test.reset_index(drop=True).astype(float)
                if len(x_frame) == len(y_series) and len(x_frame) > 0:
                    return x_frame, y_series.astype(int)
    finally:
        os.chdir(original_cwd)

    available = sorted(name for name in namespace if name.startswith(("X_", "y_")))
    raise RuntimeError(
        "Could not reconstruct an X_test/y_test pair matching feature_columns.pkl. "
        f"Available split variables: {available}"
    )


def _finite_unit_interval(values: np.ndarray, label: str) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains missing or infinite values")
    if np.min(values) < 0 or np.max(values) > 1:
        raise ValueError(f"{label} contains values outside [0, 1]")


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(value), 8) for value in values]


def export_curves() -> dict[str, Any]:
    feature_columns = list(joblib.load(ROOT / "models" / "feature_columns.pkl"))
    scaler = joblib.load(ROOT / "models" / "scaler.pkl")
    x_test, y_test = reconstruct_test_split(feature_columns)

    if set(y_test.unique()) != {0, 1}:
        raise ValueError(f"Expected binary y_test values, received {sorted(y_test.unique())}")

    output: dict[str, Any] = {
        "source": "IBM Telco Customer Churn held-out test split",
        "test_records": int(len(y_test)),
        "positive_records": int(y_test.sum()),
        "positive_rate": round(float(y_test.mean()), 8),
        "models": {},
    }

    for model_name, model_path in MODEL_PATHS.items():
        model = joblib.load(model_path)
        model_input = scaler.transform(x_test) if model_name == "Logistic Regression" else x_test.to_numpy(dtype=float)
        probabilities = np.asarray(model.predict_proba(model_input)[:, 1], dtype=float)
        _finite_unit_interval(probabilities, f"{model_name} probabilities")

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        _finite_unit_interval(fpr, f"{model_name} false-positive rates")
        _finite_unit_interval(tpr, f"{model_name} true-positive rates")
        _finite_unit_interval(precision, f"{model_name} precision")
        _finite_unit_interval(recall, f"{model_name} recall")

        roc_auc = float(roc_auc_score(y_test, probabilities))
        average_precision = float(average_precision_score(y_test, probabilities))
        if not (math.isfinite(roc_auc) and math.isfinite(average_precision)):
            raise ValueError(f"{model_name} produced invalid evaluation metrics")

        output["models"][model_name] = {
            "roc_auc": round(roc_auc, 8),
            "average_precision": round(average_precision, 8),
            "false_positive_rate": _rounded(fpr),
            "true_positive_rate": _rounded(tpr),
            "precision": _rounded(precision),
            "recall": _rounded(recall),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"Exported {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Held-out records: {output['test_records']}")
    for name, metrics in output["models"].items():
        print(
            f"{name}: ROC AUC={metrics['roc_auc']:.4f}, "
            f"Average Precision={metrics['average_precision']:.4f}"
        )
    return output


if __name__ == "__main__":
    export_curves()
