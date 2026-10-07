from flask import Flask, jsonify, render_template, request

from services.classifier import TechnicalContentClassifier
from services.keywords import extract_keywords

app = Flask(__name__)
classifier = TechnicalContentClassifier()


def analyze_content(title: str, text: str) -> dict:
    combined = f"{title} {text}".strip()
    prediction = classifier.predict(title, text)
    return {
        "categoria": prediction["categoria"],
        "probabilidad": round(prediction["probabilidad"], 4),
        "palabras_clave": extract_keywords(combined, classifier.vectorizer),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analizar")
def analyze_form():
    title = request.form.get("titulo", "").strip()
    text = request.form.get("texto", "").strip()
    if not title and not text:
        return render_template("index.html", error="Ingresa un título o contenido técnico."), 400
    return render_template("resultado.html", resultado=analyze_content(title, text), titulo=title)


@app.post("/api/contenido")
def analyze_api():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "El cuerpo JSON debe ser un objeto."}), 400
    raw_title = payload.get("titulo", "")
    raw_text = payload.get("texto", "")
    if not isinstance(raw_title, str) or not isinstance(raw_text, str):
        return jsonify({"error": "Los campos 'titulo' y 'texto' deben ser texto."}), 400
    title = raw_title.strip()
    text = raw_text.strip()
    if not title and not text:
        return jsonify({"error": "Los campos 'titulo' o 'texto' son requeridos."}), 400
    return jsonify(analyze_content(title, text))


if __name__ == "__main__":
    app.run(debug=True)

