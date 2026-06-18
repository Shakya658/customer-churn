import json
from pathlib import Path

import numpy as np

CURVE_PATH = Path(__file__).resolve().parents[1] / "assets" / "evaluation" / "model_curves.json"
EXPECTED_MODELS = {"Logistic Regression", "Random Forest", "XGBoost"}


def test_real_curve_artefact_is_valid():
    assert CURVE_PATH.exists(), "Run scripts/export_evaluation_curves.py first"
    data = json.loads(CURVE_PATH.read_text(encoding="utf-8"))

    assert data["source"] == "IBM Telco Customer Churn held-out test split"
    assert data["test_records"] > 0
    assert 0 < data["positive_records"] < data["test_records"]
    assert 0 < data["positive_rate"] < 1
    assert set(data["models"]) == EXPECTED_MODELS

    for model_name, metrics in data["models"].items():
        assert 0.5 <= metrics["roc_auc"] <= 1, model_name
        assert 0 < metrics["average_precision"] <= 1, model_name

        fpr = np.asarray(metrics["false_positive_rate"], dtype=float)
        tpr = np.asarray(metrics["true_positive_rate"], dtype=float)
        precision = np.asarray(metrics["precision"], dtype=float)
        recall = np.asarray(metrics["recall"], dtype=float)

        assert fpr.ndim == tpr.ndim == precision.ndim == recall.ndim == 1
        assert len(fpr) == len(tpr) > 1
        assert len(precision) == len(recall) > 1

        for values in (fpr, tpr, precision, recall):
            assert np.all(np.isfinite(values)), model_name
            assert np.all((0 <= values) & (values <= 1)), model_name

        assert np.all(np.diff(fpr) >= 0), model_name
        assert np.all(np.diff(tpr) >= 0), model_name
        assert np.all(np.diff(recall) <= 0), model_name
