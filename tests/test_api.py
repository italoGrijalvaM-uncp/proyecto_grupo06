"""Pruebas automáticas para la API RESTful de FastAPI."""
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_endpoint_salud():
    """Prueba el endpoint de salud /salud."""
    response = client.get("/salud")
    assert response.status_code == 200
    assert response.json().get("estado") == "ok"


def test_prediccion_valida_57_features():
    """Prueba que el endpoint /predecir responda HTTP 200 con 57 variables."""
    payload = {"features": [0.0] * 57}
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "etiqueta" in data
    assert "probabilidad_spam" in data


def test_error_400_features_incorrectas():
    """Prueba el control de errores HTTP 400 cuando no se envían 57 variables."""
    payload = {"features": [0.0, 1.0]}  # Longitud inválida (2 en vez de 57)
    response = client.post("/predecir", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()
