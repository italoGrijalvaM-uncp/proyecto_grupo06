"""Pruebas automáticas para la carga y estructura del dataset Spambase."""
import os
import pandas as pd
import pytest

RUTA_DATOS = "datos/spambase.data"


def test_existencia_archivo_datos():
    """Verifica que el archivo de datos spambase.data exista."""
    assert os.path.exists(
        RUTA_DATOS), f"El archivo {RUTA_DATOS} no fue encontrado."


def test_estructura_columnas_spambase():
    """Verifica que el dataset contenga exactamente 58 columnas (57 features + 1 target)."""
    df = pd.read_csv(RUTA_DATOS, header=None)
    assert df.shape[1] == 58, f"Se esperaban 58 columnas pero se obtuvieron {df.shape[1]}"
    assert not df.isnull().values.any(), "El dataset contiene valores nulos no permitidos."
