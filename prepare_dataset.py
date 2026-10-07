"""Prepara una muestra balanceada del dataset original de ArXiv.

Uso:
    python prepare_dataset.py data/arxiv.csv
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "arxiv.csv"
DEFAULT_OUTPUT = ROOT / "data" / "arxiv_cs_clean.csv"
RANDOM_STATE = 42
CATEGORY_MAP = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.SE": "Software Engineering",
    "cs.CR": "Information Security",
    "cs.CY": "Information Security",
    "cs.DB": "Databases",
    "cs.NI": "Networks",
}


def prepare(
    source: str | Path = DEFAULT_SOURCE,
    output: str | Path = DEFAULT_OUTPUT,
    per_category: int = 2500,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {"title", "abstract", "primary_category"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")

    original_counts = frame["primary_category"].value_counts(dropna=False)
    print("Categorías originales:")
    print(original_counts.to_string())
    frame = frame.dropna(subset=["title", "abstract", "primary_category"]).copy()
    frame = frame.drop_duplicates(subset=["title", "abstract"])
    frame["categoria"] = frame["primary_category"].map(CATEGORY_MAP)
    excluded = frame[frame["categoria"].isna()]["primary_category"].value_counts()
    print("Categorías excluidas:")
    print(excluded.to_string() if not excluded.empty else "Ninguna")

    clean = frame.dropna(subset=["categoria"])[["title", "abstract", "categoria"]].rename(columns={"title": "titulo", "abstract": "texto"})
    clean = clean.groupby("categoria", group_keys=False).apply(
        lambda group: group.sample(n=min(len(group), per_category), random_state=RANDOM_STATE)
    ).reset_index(drop=True)
    if clean[["titulo", "texto", "categoria"]].isna().any().any():
        raise ValueError("El dataset limpio contiene valores nulos")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    report = {
        "source": str(source),
        "output": str(output_path),
        "random_state": RANDOM_STATE,
        "per_category_limit": per_category,
        "rows": int(len(clean)),
        "category_counts": {str(key): int(value) for key, value in clean["categoria"].value_counts().items()},
        "excluded_categories": {str(key): int(value) for key, value in excluded.items()},
    }
    report_path = Path(report_path) if report_path is not None else ROOT / "reports" / "eda" / "preparation_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registros guardados: {len(clean)} en {output_path}")
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-category", type=int, default=2500)
    args = parser.parse_args()
    prepare(args.source, args.output, args.per_category)
