"""Entrena y exporta el modelo TF-IDF + Regresión Logística."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "arxiv_cs_clean.csv"
MODEL_DIR = ROOT / "models"


def train(data_path: str | Path = DATA_PATH, model_dir: str | Path = MODEL_DIR) -> dict:
    frame = pd.read_csv(data_path).dropna(subset=["titulo", "texto", "categoria"])
    if frame["categoria"].nunique() < 2:
        raise ValueError("Se necesitan al menos dos categorías para entrenar el modelo")
    frame["contenido"] = frame["titulo"].astype(str) + " " + frame["texto"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["contenido"], frame["categoria"], test_size=0.2, random_state=42, stratify=frame["categoria"]
    )
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000)
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train_matrix, y_train)
    predictions = model.predict(test_matrix)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    print(classification_report(y_test, predictions, zero_division=0))
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "modelo.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizador.joblib")
    return metrics


if __name__ == "__main__":
    train()

