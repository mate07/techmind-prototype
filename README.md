# TechMind Prototype

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![SweetAlert2](https://img.shields.io/badge/SweetAlert2-11-8A2BE2)](https://sweetalert2.github.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

Prototipo en Python + Flask para clasificar contenido técnico, calcular una probabilidad de confianza y extraer palabras clave mediante TF-IDF. La interfaz utiliza Bootstrap 5 y SweetAlert2 mediante CDN.

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Este único archivo incluye las dependencias de Flask, procesamiento de datos, entrenamiento, EDA y notebooks.

## Dataset, EDA y entrenamiento

El dataset original debe estar en `data/arxiv.csv` y contener `paper_id`, `title`, `abstract`, `year`, `primary_category` y `categories`.

Ejecutar el EDA completo:

```bash
python scripts/eda.py
```

El EDA analiza todo el dataset y guarda tablas, gráficos y `summary.json` en `reports/eda/`. Para los gráficos y términos frecuentes utiliza una muestra reproducible de 20,000 filas. El tamaño puede cambiarse con `--sample-size`.

Preparar el dataset para entrenamiento:

```bash
python prepare_dataset.py data/arxiv.csv
```

El script elimina nulos y duplicados, reporta categorías excluidas y crea una muestra balanceada de hasta 2,500 registros por clase en `data/arxiv_cs_clean.csv`. Las clases son Artificial Intelligence, Machine Learning, Computer Vision, Software Engineering, Information Security, Databases y Networks.

Entrenar y exportar los artefactos:

```bash
python train_model.py
```

También se puede ejecutar `notebooks/01_entrenamiento_modelo.ipynb`. Se generarán `models/modelo.joblib` y `models/vectorizador.joblib`.

## Ejecutar

```bash
python app.py
```

Abrir `http://127.0.0.1:5000`. Si todavía no existen los artefactos entrenados, la aplicación utiliza un clasificador de respaldo por reglas para permitir validar la interfaz.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/contenido \
  -H "Content-Type: application/json" \
  -d '{"titulo":"Introducción a Spring Boot","texto":"Desarrollo de APIs REST utilizando Java"}'
```

La respuesta contiene `categoria`, `probabilidad` y `palabras_clave`.

## Pruebas

```bash
pip install -r requirements-dev.txt
playwright install chromium

# Suite rápida
pytest -m "not model and not e2e"

# Preparación, entrenamiento y calidad estadística
pytest -m model

# Flask + Chromium headless
pytest -m e2e

# Validación integral
pytest
```

Las pruebas escriben datasets, reportes y modelos únicamente en directorios temporales. La suite `model` utiliza `data/arxiv_cs_clean.csv` cuando está disponible y compara sus métricas con `tests/model_baseline.json`; el archivo original `data/arxiv.csv` no es obligatorio.

La versión 0.1 no incluye autenticación, base de datos, Docker, OCI, recomendaciones ni búsqueda semántica.
