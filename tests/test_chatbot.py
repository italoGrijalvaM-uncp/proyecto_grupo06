"""Pruebas automáticas para las funciones del chatbot y la traducción/heurística."""
from src.chatbot import calcular_caracteristicas_spambase_traducido


def test_extraccion_caracteristicas_texto():
    """Verifica que el procesamiento de un correo genere un vector de 57 características."""
    correo_prueba = "¡OFERTA IMPERDIBLE! Gana un iPhone 16 Gratis HOY. Haz clic en el enlace."
    vector, senales = calcular_caracteristicas_spambase_traducido(
        correo_prueba)

    assert len(
        vector) == 57, "El vector resultante debe contener exactamente 57 características."
    assert len(
        senales) > 0, "Debe detectar señales de alerta para palabras clave o exclamaciones."
