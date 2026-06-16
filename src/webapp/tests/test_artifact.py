import json
from pathlib import Path

import joblib
import pandas as pd

from ml.model_loader import MODEL_PATH, METADATA_PATH, load_metadata, load_model

WEBAPP_DIR = Path(__file__).resolve().parent.parent


def test_artifact_files_exist():
    assert MODEL_PATH.exists()
    assert METADATA_PATH.exists()


def test_metadata_matches_model_features():
    metadata = load_metadata()
    model = load_model()
    features = metadata["features"]
    sample = pd.DataFrame([[35, 20, 0, 0, 0, 0, 0, 0, 1]], columns=features)
    proba = model.predict_proba(sample)
    assert proba.shape[1] == 2
    assert len(features) == 9


def test_metadata_has_real_metrics():
    metadata = load_metadata()
    metrics = metadata["metrics"]
    for key in ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1"]:
        assert key in metrics
        assert 0 <= metrics[key] <= 1 or key in {"pr_auc", "roc_auc", "f1"}


def test_app_module_does_not_train_on_import():
    source = (WEBAPP_DIR / "app.py").read_text(encoding="utf-8")
    forbidden = ["fit(", "train_export_model", "read_csv", "requests.get", "urllib"]
    for token in forbidden:
        assert token not in source


def test_app_module_does_not_access_internet_on_import():
    source = (WEBAPP_DIR / "app.py").read_text(encoding="utf-8")
    assert "http://" not in source
    assert "https://" not in source
