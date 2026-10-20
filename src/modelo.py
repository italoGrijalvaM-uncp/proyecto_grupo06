"""
Módulo src/modelo.py
--------------------
Construye, entrena y guarda la línea base trivial, la regresión logística
y las dos versiones de redes neuronales (MLP v1 y MLP v2) utilizando
estrictamente el conjunto de entrenamiento procesado (sin fuga).

Autor: Participante 2 / Grupo 6 - Tecnologías Emergentes
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"
DATOS_DIR = BASE_DIR / "datos"
MODELOS_DIR.mkdir(exist_ok=True, parents=True)


def entrenar_modelos():
    """
    Carga los datos y el preprocesador desde datos/, entrena los 4 modelos exigidos
    y los serializa en la carpeta modelos/.
    """
    print("=== Iniciando Etapa E3: Modelado ===")

    # 1. Cargar particiones y preprocesador generado en E2
    # Nota: Asumimos que src/datos.py ya generó los archivos CSV o podemos importarlos.
    # Para mantener la independencia, cargamos los índices o reutilizamos el script anterior.
    from datos import (
        descargar_y_cargar_datos,
        obtener_particiones_sin_fuga,
        construir_y_ajustar_preprocesador,
    )

    df = descargar_y_cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = obtener_particiones_sin_fuga(df)
    preprocesador, X_train_p = construir_y_ajustar_preprocesador(X_train)

    # Transformar validación con las reglas aprendidas exclusivamente en train
    X_val_p = preprocesador.transform(X_val)

    # 2. Definición de los 4 modelos obligatorios
    modelos = {
        # 1. Línea base trivial
        "dummy_baseline": DummyClassifier(strategy="most_frequent", random_state=42),
        # 2. Línea base simple (Modelo Lineal)
        "regresion_logistica": LogisticRegression(max_iter=1000, random_state=42),
        # 3. Red Neuronal Versión 1 (Arquitectura simple: 1 capa oculta de 50 neuronas)
        "mlp_v1": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=300,
            activation="relu",
            solver="adam",
            random_state=42,
        ),
        # 4. Red Neuronal Versión 2 (Arquitectura robusta: 2 capas ocultas, early stopping y regularización L2)
        "mlp_v2": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            alpha=0.001,  # Regularización L2
            early_stopping=True,  # Parada temprana para evitar sobreajuste
            validation_fraction=0.1,
            random_state=42,
        ),
    }

    resultados_entrenamiento = {}

    # 3. Entrenamiento y evaluación rápida en validación
    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}...")
        modelo.fit(X_train_p, y_train)

        # Predicción en validación para verificación inicial
        preds_val = modelo.predict(X_val_p)
        f1_val = f1_score(y_val, preds_val, zero_division=0)

        print(f"-> {nombre} entrenado con éxito. F1-Score en Validación: {f1_val:.4f}")

        # Guardar el modelo entrenado en disco
        ruta_modelo = MODELOS_DIR / f"{nombre}.joblib"
        joblib.dump(modelo, ruta_modelo)

        resultados_entrenamiento[nombre] = {"modelo": modelo, "f1_val": f1_val}

    print(
        "\n¡Todos los modelos de la Etapa E3 han sido entrenados y guardados en 'modelos/'!"
    )
    return resultados_entrenamiento


if __name__ == "__main__":
    entrenar_modelos()
