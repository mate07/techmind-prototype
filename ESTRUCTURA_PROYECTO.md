# Estructura del proyecto TechMind Prototype

Revisión: 5 de octubre de 2026. Resumen basado en los archivos y el código presentes en el proyecto.

## 1. Visión general

TechMind es un prototipo de clasificación de contenido técnico desarrollado en Python con Flask. Recibe un título, un texto o ambos y devuelve una categoría, un valor de confianza y hasta cinco palabras clave. Ofrece un formulario web y una API JSON que comparten la misma lógica de análisis.

La solución tiene cuatro componentes principales: interfaz web, servicios de inferencia, preparación y entrenamiento del modelo, y análisis exploratorio de datos (EDA). El modelo implementado en el código de entrenamiento combina TF-IDF y regresión logística. La clasificación devuelve una sola categoría, aunque el dataset de origen contiene información multietiqueta.

## 2. Árbol de carpetas y archivos

```text
techmind-prototype/
├── app.py
├── prepare_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── plan.md
├── ESTRUCTURA_PROYECTO.md
├── .gitignore
├── data/
│   └── arxiv_cs_clean.csv
├── models/
│   ├── modelo.joblib
│   └── vectorizador.joblib
├── services/
│   ├── __init__.py
│   ├── classifier.py
│   └── keywords.py
├── templates/
│   ├── index.html
│   └── resultado.html
├── static/
│   └── styles.css
├── scripts/
│   ├── __init__.py
│   └── eda.py
├── notebooks/
│   ├── 00_eda_dataset.ipynb
│   └── 01_entrenamiento_modelo.ipynb
├── reports/
│   ├── eda/
│   └── eda-compat-check/
├── tests/
│   └── test_app.py
├── .agents/
├── .git/
└── .venv/
```

El árbol resume el código y los artefactos relevantes; omite el contenido interno del entorno virtual y de Git. La carpeta `.agents/` está vacía en la revisión. El archivo original `data/arxiv.csv`, esperado por el flujo de EDA y preparación, no está presente en esta copia local.

## 3. Aplicación y servicios

### `app.py`: entrada de la aplicación

Crea la aplicación Flask y una instancia de `TechnicalContentClassifier` al importar el módulo. La función `analyze_content(title, text)` coordina clasificación y extracción de palabras clave, y redondea la probabilidad a cuatro decimales.

| Método y ruta | Entrada | Función y salida |
| --- | --- | --- |
| `GET /` | Sin cuerpo | Renderiza el formulario de `index.html`. |
| `POST /analizar` | Campos de formulario `titulo` y `texto` | Analiza el contenido y renderiza `resultado.html`. |
| `POST /api/contenido` | Objeto JSON con `titulo` y/o `texto` | Devuelve `categoria`, `probabilidad` y `palabras_clave`. |

Ambas rutas de análisis permiten proporcionar solo uno de los campos. Cuando ambos están vacíos, responden con HTTP 400. Al ejecutar `python app.py`, se inicia el servidor de desarrollo con `debug=True`.

Ejemplo de entrada de la API:

```json
{
  "titulo": "Introducción a Spring Boot",
  "texto": "Desarrollo de APIs REST utilizando Java"
}
```

La salida contiene una categoría de tipo texto, una probabilidad numérica y una lista de palabras clave. Los valores concretos dependen de los artefactos cargados o del mecanismo de respaldo.

### `services/`: lógica de análisis

| Archivo | Responsabilidad |
| --- | --- |
| `classifier.py` | Localiza y carga los artefactos con Joblib; transforma el contenido, obtiene probabilidades con `predict_proba` y selecciona la clase con mayor valor. |
| `keywords.py` | Extrae hasta cinco términos con mayor peso TF-IDF; sin vectorizador, utiliza frecuencia de palabras y una lista de palabras vacías en español e inglés. |
| `__init__.py` | Define el paquete de servicios. |

Si falta el modelo o el vectorizador, la clasificación utiliza reglas de coincidencia de términos. Sin coincidencias, devuelve `Software Engineering` con confianza de `0.35`. Con coincidencias, calcula una confianza heurística limitada a `0.95`; este valor de respaldo no es una probabilidad obtenida del modelo. La ausencia de archivos activa el respaldo, pero los errores al cargar un artefacto existente no se capturan en este servicio.

### `templates/` y `static/`: presentación

- `index.html`: formulario en español para introducir título y contenido, con errores del servidor mostrados mediante SweetAlert2.
- `resultado.html`: muestra categoría, confianza en porcentaje, palabras clave y enlace para iniciar otro análisis.
- `styles.css`: define colores, fondo, tarjetas, campos de formulario y etiquetas de palabras clave.

Las páginas usan plantillas Jinja de Flask, Bootstrap 5.3.3 y SweetAlert2 11 mediante CDN. El JavaScript de las alertas está incluido en las plantillas; no existe un proyecto frontend separado ni un proceso de compilación de recursos.

## 4. Datos y entrenamiento

### `data/` y `prepare_dataset.py`

`data/arxiv_cs_clean.csv` es el dataset preparado para entrenamiento. Sus columnas de trabajo son `titulo`, `texto` y `categoria`.

`prepare_dataset.py` lee por defecto `data/arxiv.csv`, exige las columnas `title`, `abstract` y `primary_category`, elimina nulos en esos campos y duplicados por título y resumen, y transforma las categorías de origen:

| Categoría de ArXiv | Categoría del prototipo |
| --- | --- |
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.CV` | Computer Vision |
| `cs.SE` | Software Engineering |
| `cs.CR`, `cs.CY` | Information Security |
| `cs.DB` | Databases |
| `cs.NI` | Networks |

Las demás categorías se excluyen. El script toma hasta 2,500 registros por clase con semilla 42; las clases con menos registros conservan todos los disponibles, por lo que no garantiza tamaños idénticos. Genera el CSV limpio y `reports/eda/preparation_summary.json`.

Permite cambiar el origen mediante un argumento posicional, el destino con `--output` y el límite por clase con `--per-category`.

### `train_model.py`: entrenamiento reproducible

1. Lee el CSV limpio y elimina filas sin título, texto o categoría.
2. Concatena título y texto en una columna de contenido.
3. Divide los datos en 80 % para entrenamiento y 20 % para prueba, con estratificación y semilla 42.
4. Ajusta TF-IDF con minúsculas, unigramas y bigramas, `min_df=2` y hasta 50,000 características.
5. Entrena `LogisticRegression` con `max_iter=1000` y `class_weight="balanced"`.
6. Calcula accuracy, precision, recall y F1 ponderados, reporte por clase y matriz de confusión.
7. Exporta el clasificador y el vectorizador a `models/`.

La función `train()` devuelve las métricas y muestra el reporte de clasificación; el script no guarda automáticamente un archivo de métricas.

### `models/`: artefactos de inferencia

- `modelo.joblib`: artefacto del clasificador.
- `vectorizador.joblib`: artefacto de transformación TF-IDF.

Ambos archivos existen en la copia inspeccionada. Su presencia se verificó por inventario; no se cargaron ni se evaluaron para redactar este documento. Flask los carga al crear el servicio, sin volver a entrenar en cada solicitud.

## 5. Exploración y reportes

### `scripts/eda.py`

Implementa el análisis exploratorio por línea de comandos. Requiere `paper_id`, `title`, `abstract`, `year`, `primary_category` y `categories` en el CSV original.

Analiza nulos, duplicados, registros incompletos, categorías, años y longitudes de texto. Usa una muestra reproducible de hasta 20,000 filas para términos frecuentes, coocurrencia de categorías y gráficos de longitudes. Los conteos de categorías y años se calculan sobre el dataset completo.

Sus opciones son `--input`, `--output` y `--sample-size`. La salida predeterminada es `reports/eda/`.

### `notebooks/`

| Notebook | Contenido implementado |
| --- | --- |
| `00_eda_dataset.ipynb` | Inspección de datos, nulos y duplicados, longitudes, categorías y términos frecuentes; exporta tablas de categorías y estadísticas de longitud. |
| `01_entrenamiento_modelo.ipynb` | Flujo interactivo de vectorización, entrenamiento, evaluación y exportación de los dos artefactos. |

El notebook de EDA cubre parte del análisis del script, no todas sus salidas. Ambos notebooks calculan la raíz como `Path.cwd().parent`, por lo que esperan ejecutarse con `notebooks/` como directorio de trabajo.

### `reports/`

Las carpetas `eda/` y `eda-compat-check/` contienen las siguientes salidas:

| Archivo | Contenido |
| --- | --- |
| `nulls.csv` | Cantidad de nulos por columna. |
| `primary_categories.csv` y `.png` | Distribución de categorías principales. |
| `years.csv` y `.png` | Distribución temporal. |
| `text_length_stats.csv` | Estadísticas de longitud de títulos y resúmenes. |
| `text_lengths.png` | Distribución de longitudes en la muestra. |
| `top_terms.csv` y `.png` | Términos frecuentes de los resúmenes. |
| `category_cooccurrence.csv` | Frecuencia de pares de categorías en la muestra. |
| `summary.json` | Resumen de la ejecución de EDA. |

`eda/` también contiene `preparation_summary.json`. El resumen guardado en `eda-compat-check/` registra 200,094 filas y una muestra de 10 filas. Son resultados previos, no una ejecución realizada durante esta revisión; no debe confundirse esa muestra con el valor predeterminado de 20,000.

## 6. Flujo entre componentes

```text
Preparación y entrenamiento:
CSV original → prepare_dataset.py → CSV limpio → train_model.py
                                                   ↓
                                     modelo.joblib + vectorizador.joblib

Análisis de contenido:
Formulario o cliente JSON → app.py → services/classifier.py
                                 └→ services/keywords.py
                                          ↓
                           Resultado HTML o respuesta JSON

Exploración:
CSV original → scripts/eda.py → reports/eda/
```

El EDA es una etapa de inspección independiente: el script de entrenamiento consume directamente el CSV limpio, sin leer los reportes. Los notebooks permiten explorar y entrenar de forma interactiva.

## 7. Dependencias y archivos de soporte

| Elemento | Función |
| --- | --- |
| `requirements.txt` | Declara Flask `>=3.0,<4.0`, pandas `>=2.0,<3.0`, scikit-learn `>=1.3,<2.0`, Joblib `>=1.3,<2.0`, Jupyter `>=1.0,<2.0`, Matplotlib `>=3.7,<4.0` y Seaborn `>=0.12,<1.0`. Son rangos permitidos, no versiones instaladas verificadas. |
| `README.md` | Presentación, instalación, comandos de EDA, preparación, entrenamiento, uso y pruebas. Indica Python 3.10 o superior. |
| `plan.md` | Plan de trabajo y alcance previsto para el prototipo 0.1; no constituye evidencia de ejecución de cada tarea. |
| `.gitignore` | Excluye entornos, cachés, `.env`, modelos Joblib, dataset limpio y reportes de `reports/eda/`, entre otros archivos locales. |
| `.venv/` | Entorno virtual local de Python. |
| `.git/` | Metadatos del control de versiones. |
| `.agents/` | Carpeta local vacía en la inspección. |

Los datos, modelos y reportes pueden existir localmente aunque estén excluidos de Git. El proyecto no implementa autenticación, persistencia en base de datos, Docker, despliegue OCI, recomendaciones ni búsqueda semántica.

## 8. Pruebas y comandos de referencia

`tests/test_app.py` utiliza `unittest` y el cliente de pruebas de Flask. Contiene tres pruebas: carga de la página inicial, presencia de las claves de análisis en la API y rechazo de una entrada vacía. La prueba de análisis utiliza texto de ciberseguridad, pero no comprueba una categoría exacta ni mide la calidad predictiva. Los tres casos temáticos descritos en el plan no equivalen a tres pruebas automatizadas de precisión implementadas.

Desde la raíz del proyecto, con el entorno Python apropiado:

```powershell
# Instalar dependencias
pip install -r requirements.txt

# EDA y preparación: requieren suministrar data/arxiv.csv
python scripts/eda.py
python prepare_dataset.py data/arxiv.csv

# Entrenar con el CSV limpio
python train_model.py

# Iniciar la aplicación local
python app.py

# Ejecutar las pruebas existentes
python -m unittest discover -s tests -v
```

La aplicación se abre por defecto en `http://127.0.0.1:5000`. Los comandos se documentan como referencia: no se ejecutaron instalación, entrenamiento, servidor ni pruebas durante esta revisión documental.

