import json

import joblib
import pandas as pd
import pytest

from prepare_dataset import CATEGORY_MAP, RANDOM_STATE, prepare
from train_model import train


pytestmark = pytest.mark.model


def source_frame():
    rows = []
    for code in CATEGORY_MAP:
        for suffix in ("alpha", "beta", "gamma"):
            rows.append({"title": f"{code} {suffix}", "abstract": f"contenido {code} {suffix}", "primary_category": code})
    rows += [rows[0].copy(), {"title": None, "abstract": "incompleto", "primary_category": "cs.AI"},
             {"title": "fuera", "abstract": "otra disciplina", "primary_category": "math.ST"}]
    return pd.DataFrame(rows)


def test_prepare_requires_source_columns(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"title": ["x"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        prepare(source, tmp_path / "clean.csv", report_path=tmp_path / "report.json")


def test_prepare_cleans_maps_limits_and_reports_reproducibly(tmp_path):
    source = tmp_path / "source.csv"
    source_frame().to_csv(source, index=False)
    report = tmp_path / "preparation.json"
    result = prepare(source, tmp_path / "first.csv", 2, report)
    repeated = prepare(source, tmp_path / "second.csv", 2, tmp_path / "repeated.json")

    assert set(result.columns) == {"titulo", "texto", "categoria"}
    assert set(result["categoria"]) == set(CATEGORY_MAP.values())
    assert result.groupby("categoria").size().max() == 2
    assert not result.isna().any().any()
    pd.testing.assert_frame_equal(result, repeated)
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["rows"] == len(result)
    assert summary["excluded_categories"] == {"math.ST": 1}
    assert summary["random_state"] == RANDOM_STATE


def training_frame(samples_per_class=12):
    rows = []
    for label, token in [("Databases", "sql database query"), ("Networks", "tcp routing network")]:
        for index in range(samples_per_class):
            rows.append({"titulo": f"{token} {index}", "texto": f"{token} example {index}", "categoria": label})
    return pd.DataFrame(rows)


def test_train_rejects_one_class(tmp_path):
    data = tmp_path / "one-class.csv"
    training_frame().query("categoria == 'Databases'").to_csv(data, index=False)
    with pytest.raises(ValueError, match="al menos dos categorías"):
        train(data, tmp_path / "models")


def test_train_exports_compatible_artifacts_and_metrics(tmp_path):
    data = tmp_path / "training.csv"
    training_frame().to_csv(data, index=False)
    output = tmp_path / "models"
    metrics = train(data, output)
    model = joblib.load(output / "modelo.joblib")
    vectorizer = joblib.load(output / "vectorizador.joblib")
    probabilities = model.predict_proba(vectorizer.transform(["sql database query"]))[0]

    expected = {"accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "classification_report", "confusion_matrix"}
    assert expected <= set(metrics)
    assert len(metrics["confusion_matrix"]) == len(model.classes_) == 2
    assert probabilities.sum() == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in probabilities)


def test_training_is_reproducible(tmp_path):
    data = tmp_path / "training.csv"
    training_frame().to_csv(data, index=False)
    first = train(data, tmp_path / "first")
    second = train(data, tmp_path / "second")
    assert first["f1_weighted"] == second["f1_weighted"]
    assert first["confusion_matrix"] == second["confusion_matrix"]
