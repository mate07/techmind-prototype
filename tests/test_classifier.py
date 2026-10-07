from pathlib import Path

import joblib
import numpy as np
import pytest

from services.classifier import FALLBACK_RULES, TechnicalContentClassifier


pytestmark = pytest.mark.unit


class FakeVectorizer:
    def __init__(self):
        self.seen = None

    def transform(self, values):
        self.seen = values
        return values


class FakeModel:
    classes_ = np.array(["Databases", "Networks"])

    def predict_proba(self, matrix):
        return np.array([[0.2, 0.8]])


def test_predict_uses_model_and_combines_title_and_text(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    joblib.dump(FakeModel(), model_path)
    joblib.dump(FakeVectorizer(), vectorizer_path)

    classifier = TechnicalContentClassifier(model_path, vectorizer_path)
    result = classifier.predict("TCP", "routing protocol")

    assert result == {"categoria": "Networks", "probabilidad": 0.8}
    assert classifier.vectorizer.seen == ["TCP routing protocol"]


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("artificial intelligence", "Artificial Intelligence"),
        ("spring backend", "Software Engineering"),
        ("encryption vulnerability", "Information Security"),
        ("object detection", "Computer Vision"),
        ("postgresql query", "Databases"),
        ("tcp routing", "Networks"),
        ("regression training", "Machine Learning"),
    ],
)
def test_fallback_recognizes_every_category(text, category):
    assert TechnicalContentClassifier._fallback_predict(text)["categoria"] == category


def test_fallback_default_and_confidence_cap():
    assert TechnicalContentClassifier._fallback_predict("sin términos conocidos") == {
        "categoria": "Software Engineering",
        "probabilidad": 0.35,
    }
    repeated_terms = " ".join(FALLBACK_RULES["Software Engineering"])
    assert TechnicalContentClassifier._fallback_predict(repeated_terms)["probabilidad"] <= 0.95


def test_missing_artifacts_enable_fallback(tmp_path):
    classifier = TechnicalContentClassifier(tmp_path / "missing-model", tmp_path / "missing-vectorizer")
    assert classifier.model is None
    assert classifier.predict("SQL", "database query")["categoria"] == "Databases"
