import json
from pathlib import Path

import pytest

from train_model import DATA_PATH, train


pytestmark = [pytest.mark.model, pytest.mark.integration]
BASELINE_PATH = Path(__file__).with_name("model_baseline.json")


def test_real_model_meets_quality_gates(tmp_path):
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset preparado no disponible: {DATA_PATH}")

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    metrics = train(DATA_PATH, tmp_path / "models")
    report = metrics["classification_report"]
    classes = [values for name, values in report.items() if name not in {"accuracy", "macro avg", "weighted avg"}]
    required_f1 = max(baseline["minimum_f1_weighted"], baseline["f1_weighted"] - baseline["maximum_regression"])

    assert metrics["f1_weighted"] >= required_f1
    assert all(values["f1-score"] >= baseline["minimum_f1_per_class"] for values in classes)
    assert len(metrics["confusion_matrix"]) == len(classes)
    assert all(len(row) == len(classes) for row in metrics["confusion_matrix"])
