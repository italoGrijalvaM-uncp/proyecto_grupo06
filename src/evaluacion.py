"""
Módulo src/evaluacion.py
------------------------
Evalúa los modelos entrenados en E3 utilizando el conjunto de validación y prueba,
genera fichas técnicas (JSON) de versionado para cada uno, y promueve al ganador.

Autor: Grupo 6 - Tecnologías Emergentes
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos"
MODELOS_DIR = BASE_DIR / "modelos"
MODELOS_DIR.mkdir(exist_ok=True, parents=True)


def evaluar_y_versionar():
    print("=== Iniciando Etapa E4: Evaluación y Versionado ===")

    # 1. Cargar datos y preprocesador
    from datos import descargar_y_cargar_datos, obtener_particiones_sin_fuga

    df = descargar_y_cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = obtener_particiones_sin_fuga(
        df)

    # Cargar preprocesador ajustado en train
    preprocesador_path = DATOS_DIR / "preprocesador_referencia_train.joblib"
    if not preprocesador_path.exists():
        raise FileNotFoundError(
            "No se encontró el preprocesador. Ejecuta primero src/datos.py y src/modelo.py."
        )

    preprocesador = joblib.load(preprocesador_path)

    # Transformar conjuntos de validación y prueba sin fuga
    X_val_p = preprocesador.transform(X_val)
    X_test_p = preprocesador.transform(X_test)

    # Lista de modelos a evaluar
    nombres_modelos = ["dummy_baseline",
                       "regresion_logistica", "mlp_v1", "mlp_v2"]

    mejor_modelo_nombre = None
    mejor_f1_val = -1.0

    for nombre in nombres_modelos:
        ruta_modelo = MODELOS_DIR / f"{nombre}.joblib"
        if not ruta_modelo.exists():
            print(
                f"Advertencia: No se encontró el modelo {nombre}. Omitiendo...")
            continue

        modelo = joblib.load(ruta_modelo)

        # --- EVALUACIÓN EN VALIDACIÓN (Para selección) ---
        preds_val = modelo.predict(X_val_p)
        probs_val = (
            modelo.predict_proba(X_val_p)[:, 1]
            if hasattr(modelo, "predict_proba")
            else preds_val
        )

        f1_val = f1_score(y_val, preds_val, zero_division=0)
        precision_val = precision_score(y_val, preds_val, zero_division=0)
        recall_val = recall_score(y_val, preds_val, zero_division=0)
        pr_auc_val = average_precision_score(y_val, probs_val)

        # --- EVALUACIÓN EN PRUEBA (Examen final único) ---
        preds_test = modelo.predict(X_test_p)
        probs_test = (
            modelo.predict_proba(X_test_p)[:, 1]
            if hasattr(modelo, "predict_proba")
            else preds_test
        )

        f1_test = f1_score(y_test, preds_test, zero_division=0)
        precision_test = precision_score(y_test, preds_test, zero_division=0)
        recall_test = recall_score(y_test, preds_test, zero_division=0)
        pr_auc_test = average_precision_score(y_test, probs_test)

        # 2. Construir Ficha de Versionado (JSON)
        ficha = {
            "nombre_modelo": nombre,
            "fecha_evaluacion_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "UCI Spambase",
            "parametros": str(
                modelo.get_params() if hasattr(modelo, "get_params") else "N/A"
            ),
            "metricas_validacion": {
                "f1_score": round(float(f1_val), 4),
                "precision": round(float(precision_val), 4),
                "recall": round(float(recall_val), 4),
                "pr_auc": round(float(pr_auc_val), 4),
            },
            "metricas_prueba": {
                "f1_score": round(float(f1_test), 4),
                "precision": round(float(precision_test), 4),
                "recall": round(float(recall_test), 4),
                "pr_auc": round(float(pr_auc_test), 4),
            },
        }

        # Guardar JSON individual por versión
        ruta_json = MODELOS_DIR / f"ficha_{nombre}.json"
        ruta_json.write_text(
            json.dumps(ficha, indent=4, ensure_ascii=False), encoding="utf-8"
        )

        print(
            f"[{nombre}] -> F1 Val: {f1_val:.4f} | F1 Test: {f1_test:.4f} (Ficha guardada)"
        )

        # 3. Selección del ganador basada EXCLUSIVAMENTE en Validación
        if f1_val > mejor_f1_val:
            mejor_f1_val = f1_val
            mejor_modelo_nombre = nombre

    # 4. Promoción de la versión elegida
    if mejor_modelo_nombre:
        print(
            f"\n🏆 ¡Modelo Ganador Seleccionado (por Validación): {mejor_modelo_nombre} (F1: {mejor_f1_val:.4f})!"
        )

        # Copiar/Guardar el ganador como el modelo oficial de producción
        modelo_ganador = joblib.load(
            MODELOS_DIR / f"{mejor_modelo_nombre}.joblib")
        joblib.dump(modelo_ganador, MODELOS_DIR / "modelo_produccion.joblib")

        promocion_info = {
            "modelo_promovido": mejor_modelo_nombre,
            "criterio_seleccion": "Mayor F1-Score en conjunto de validación",
            "fecha_promocion_utc": datetime.now(timezone.utc).isoformat(),
            "archivo_produccion": "modelo_produccion.joblib",
        }
        (MODELOS_DIR / "promocion_ganador.json").write_text(
            json.dumps(promocion_info, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        print("Modelo promovido guardado exitosamente como 'modelo_produccion.joblib'.")


if __name__ == "__main__":
    evaluar_y_versionar()
