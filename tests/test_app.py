import pytest


pytestmark = pytest.mark.integration


class StubClassifier:
    vectorizer = None

    def predict(self, title, text):
        return {"categoria": "Software Engineering", "probabilidad": 0.87654}


@pytest.fixture(autouse=True)
def stub_analysis(monkeypatch):
    monkeypatch.setattr("app.classifier", StubClassifier())


def test_home_renders_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Analiza contenido" in response.data


@pytest.mark.parametrize("payload", [
    {"titulo": "API REST", "texto": ""},
    {"titulo": "", "texto": "Desarrollo de software"},
    {"titulo": "API REST", "texto": "Desarrollo de software"},
])
def test_api_accepts_title_or_text(client, payload):
    response = client.post("/api/contenido", json=payload)
    body = response.get_json()
    assert response.status_code == 200
    assert set(body) == {"categoria", "probabilidad", "palabras_clave"}
    assert isinstance(body["categoria"], str)
    assert isinstance(body["probabilidad"], float) and 0 <= body["probabilidad"] <= 1
    assert isinstance(body["palabras_clave"], list)


@pytest.mark.parametrize("payload", [{}, {"titulo": "  ", "texto": "\n"}])
def test_api_rejects_empty_content(client, payload):
    response = client.post("/api/contenido", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


@pytest.mark.parametrize("payload", [[], {"titulo": 123, "texto": "válido"}, {"titulo": "válido", "texto": {}}])
def test_api_rejects_non_text_values(client, payload):
    assert client.post("/api/contenido", json=payload).status_code == 400


def test_api_rejects_null_and_malformed_json(client):
    assert client.post("/api/contenido", json=None).status_code == 400
    assert client.post("/api/contenido", data="{", content_type="application/json").status_code == 400


@pytest.mark.parametrize("form", [
    {"titulo": "Solo título", "texto": ""},
    {"titulo": "", "texto": "Solo contenido"},
    {"titulo": "Título", "texto": "Contenido"},
])
def test_form_renders_result(client, form):
    response = client.post("/analizar", data=form)
    assert response.status_code == 200
    assert b"Software Engineering" in response.data
    assert b"87.6%" in response.data


def test_form_rejects_blank_content(client):
    response = client.post("/analizar", data={"titulo": " ", "texto": " "})
    assert response.status_code == 400
    assert b"server-error" in response.data


def test_internal_failure_is_not_reported_as_success(client, monkeypatch):
    class BrokenClassifier:
        vectorizer = None
        def predict(self, title, text):
            raise RuntimeError("fallo controlado")

    monkeypatch.setattr("app.classifier", BrokenClassifier())
    with pytest.raises(RuntimeError, match="fallo controlado"):
        client.post("/api/contenido", json={"titulo": "API", "texto": "REST"})
