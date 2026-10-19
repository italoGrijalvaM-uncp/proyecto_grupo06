"""
Módulo src/datos.py
-------------------
Encargado de la carga, auditoría de calidad, partición estratificada por grupos
y generación del preprocesador sin fuga de información para el dataset UCI Spambase.

Autor: Participante 1 (Grupo 6 - Tecnologías Emergentes)
"""

import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos"
RESULTADOS_DIR = BASE_DIR / "resultados_eda"
DATOS_DIR.mkdir(exist_ok=True, parents=True)
RESULTADOS_DIR.mkdir(exist_ok=True, parents=True)

# 57 Características numéricas estadísticas + 1 etiqueta binaria
WORDS = [
    "make",
    "address",
    "all",
    "3d",
    "our",
    "over",
    "remove",
    "internet",
    "order",
    "mail",
    "receive",
    "will",
    "people",
    "report",
    "addresses",
    "free",
    "business",
    "email",
    "you",
    "credit",
    "your",
    "font",
    "000",
    "money",
    "hp",
    "hpl",
    "george",
    "650",
    "lab",
    "labs",
    "telnet",
    "857",
    "data",
    "415",
    "85",
    "technology",
    "1999",
    "parts",
    "pm",
    "direct",
    "cs",
    "meeting",
    "original",
    "project",
    "re",
    "edu",
    "table",
    "conference",
]
CHARS = [";", "(", "[", "!", "$", "#"]
FEATURES = (
    [f"word_freq_{w}" for w in WORDS]
    + [f"char_freq_{ch}" for ch in CHARS]
    + [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
)
COLUMNAS = FEATURES + ["spam"]


def descargar_y_cargar_datos() -> pd.DataFrame:
    """
    Descarga el dataset Spambase desde el repositorio oficial UCI si no existe localmente,
    valida la cantidad de columnas y retorna el DataFrame con los nombres asignados.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    ruta_archivo = DATOS_DIR / "spambase.data"

    if not ruta_archivo.exists():
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            ruta_archivo.write_bytes(r.content)
            print("Dataset descargado exitosamente desde UCI.")
        except requests.RequestException as exc:
            raise FileNotFoundError(
                f"No se pudo descargar el dataset. Guarda manualmente 'spambase.data' en {DATOS_DIR}"
            ) from exc

    df = pd.read_csv(ruta_archivo, header=None, names=COLUMNAS)

    if df.shape[1] != 58:
        raise ValueError(
            f"Se esperaban 58 columnas, pero se encontraron {df.shape[1]}."
        )

    return df


def auditardatos(df: pd.DataFrame) -> dict:
    """
    Realiza la auditoría de calidad: tipos, valores nulos, dominios y duplicados.
    """
    X_num = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    no_numericos = int(X_num.isna().sum().sum() - df[FEATURES].isna().sum().sum())
    negativos = int((X_num < 0).sum().sum())
    obj_invalido = int((~df["spam"].isin([0, 1])).sum())
    duplicados_exactos = int(df.duplicated().sum())
    duplicados_X = int(df.duplicated(subset=FEATURES).sum())

    if no_numericos or negativos or obj_invalido:
        raise ValueError(
            "Existen registros fuera del dominio válido (valores negativos o no numéricos)."
        )

    reporte = {
        "total_filas": len(df),
        "total_features": len(FEATURES),
        "celdas_nulas": int(df.isna().sum().sum()),
        "duplicados_exactos": duplicados_exactos,
        "duplicados_X": duplicados_X,
    }
    return reporte


def obtener_particiones_sin_fuga(df: pd.DataFrame, seed_outer=42, seed_inner=43):
    """
    Aplica una partición estratificada por grupos de vectores idénticos (StratifiedGroupKFold)
    en proporciones aproximadas 60% Train, 20% Val, 20% Test para prevenir la fuga de datos.

    Retorna X_train, X_val, X_test, y_train, y_val, y_test e índices originales.
    """
    # Identificar vectores de características idénticos mediante hashing
    huellas = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()
    y = df["spam"].astype(int).to_numpy()

    # Partición externa (Test ~20%)
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed_outer)
    idx_dev, idx_test = next(sgkf_outer.split(df[FEATURES], y, groups=huellas))

    # Partición interna sobre dev (Train ~60%, Val ~20%)
    resto = df.iloc[idx_dev]
    hue_resto = huellas[idx_dev]
    y_resto = y[idx_dev]

    sgkf_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed_inner)
    local_train, local_val = next(
        sgkf_inner.split(resto[FEATURES], y_resto, groups=hue_resto)
    )

    idx_train = idx_dev[local_train]
    idx_val = idx_dev[local_val]

    # Guardar índices reproducibles para todo el equipo
    pd.DataFrame({"indice_original": idx_train}).to_csv(
        DATOS_DIR / "indices_train.csv", index=False
    )
    pd.DataFrame({"indice_original": idx_val}).to_csv(
        DATOS_DIR / "indices_val.csv", index=False
    )
    pd.DataFrame({"indice_original": idx_test}).to_csv(
        DATOS_DIR / "indices_test.csv", index=False
    )

    X_train = df.iloc[idx_train][FEATURES]
    X_val = df.iloc[idx_val][FEATURES]
    X_test = df.iloc[idx_test][FEATURES]

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    return X_train, X_val, X_test, y_train, y_val, y_test


def construir_y_ajustar_preprocesador(X_train: pd.DataFrame):
    """
    Crea un Pipeline con SimpleImputer y StandardScaler,
    y lo ajusta EXCLUSIVAMENTE con el conjunto de entrenamiento (fit_transform).
    """
    preprocesador = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ]
    )

    # Fit solo en entrenamiento
    X_train_p = preprocesador.fit_transform(X_train)

    # Guardar objeto preprocesador para su reutilización en API/Modelado
    joblib.dump(preprocesador, DATOS_DIR / "preprocesador_referencia_train.joblib")

    return preprocesador, X_train_p


if __name__ == "__main__":
    print("=== Ejecutando flujo automatizado de datos (src/datos.py) ===")
    df_data = descargar_y_cargar_datos()
    reporte_calidad = auditardatos(df_data)
    print("Reporte de calidad:", reporte_calidad)

    X_tr, X_v, X_te, y_tr, y_v, y_te = obtener_particiones_sin_fuga(df_data)
    print(
        f"Particiones generadas -> Train: {len(X_tr)}, Val: {len(X_v)}, Test: {len(X_te)}"
    )

    prep, X_tr_p = construir_y_ajustar_preprocesador(X_tr)
    print("Preprocesador ajustado sobre Train y guardado exitosamente.")
