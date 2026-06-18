from pathlib import Path

import joblib


MODEL_FILES = {
    "Logistic Regression": Path("models/logistic_model.pkl"),
    "Random Forest": Path("models/random_forest_model.pkl"),
    "XGBoost": Path("models/xgb_model.pkl"),
    "Scaler": Path("models/scaler.pkl"),
    "Feature columns": Path("models/feature_columns.pkl"),
    "Thresholds": Path("models/thresholds.pkl"),
}


def test_model_artefacts_exist_and_load():
    loaded = {}

    for label, path in MODEL_FILES.items():
        assert path.exists(), f"Missing {label} artefact: {path}"
        loaded[label] = joblib.load(path)

    feature_columns = loaded["Feature columns"]
    thresholds = loaded["Thresholds"]

    assert len(feature_columns) > 0, "Saved feature list is empty"
    assert isinstance(thresholds, dict), "Thresholds artefact must be a dictionary"

    expected_models = {"Logistic Regression", "Random Forest", "XGBoost"}
    assert expected_models.issubset(thresholds), (
        "Thresholds are missing one or more model keys: "
        f"{sorted(expected_models - set(thresholds))}"
    )

    for model_name in expected_models:
        threshold = thresholds[model_name]
        assert 0 <= threshold <= 1, f"Invalid threshold for {model_name}: {threshold}"
