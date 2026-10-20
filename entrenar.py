"""
Script Principal de Entrenamiento (entrenar.py)
Orquesta la carga de datos, el entrenamiento de la línea base y las dos versiones de MLP.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from src.modelo import construir_linea_base, construir_red_v1, construir_red_v2, obtener_callbacks

# Crear directorio de modelos si no existe
os.makedirs("modelos", exist_ok=True)


def ejecutar_entrenamiento():
    print("=== 1. Carga de datos de prueba / simulación para Spambase ===")
    
    # Intenta importar la función de carga del Integrante 1 si ya existe en src/datos.py
    try:
        from src.datos import cargar_y_preparar_datos
        X_train, X_val, X_test, y_train, y_val, y_test, escalador = cargar_y_preparar_datos()
        print("Datos cargados correctamente desde src/datos.py")
    except Exception as e:
        print("Aviso: src/datos.py aún no está listo o completo. Generando datos sintéticos de 57 características para validación...")
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X, y = make_classification(n_samples=1000, n_features=57, n_informative=40, random_state=42)
        X_train_raw, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y)

        escalador = StandardScaler()
        X_train = escalador.fit_transform(X_train_raw)
        X_val = escalador.transform(X_val_raw)
        X_test = escalador.transform(X_test_raw)

    # Guardar el escalador ajustado
    with open("modelos/escalador.pkl", "wb") as f:
        pickle.dump(escalador, f)

    input_dim = X_train.shape[1]

    print("\n=== 2. Entrenando Línea Base (Regresión Logística) ===")
    baseline = construir_linea_base()
    baseline.fit(X_train, y_train)
    pred_base_val = baseline.predict(X_val)
    f1_base = f1_score(y_val, pred_base_val)
    print(f"Línea Base -> F1-Score en Validación: {f1_base:.4f}")

    with open("modelos/baseline_logistica.pkl", "wb") as f:
        pickle.dump(baseline, f)

    print("\n=== 3. Entrenando Red Neuronal Versión 1 (MLP Básico) ===")
    red_v1 = construir_red_v1(input_dim=input_dim)
    red_v1.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=obtener_callbacks(),
        verbose=1
    )
    prob_v1_val = red_v1.predict(X_val, verbose=0).flatten()
    pred_v1_val = (prob_v1_val > 0.5).astype(int)
    f1_v1 = f1_score(y_val, pred_v1_val)
    print(f"Red V1 -> F1-Score en Validación: {f1_v1:.4f}")
    red_v1.save("modelos/red_v1.keras")

    print("\n=== 4. Entrenando Red Neuronal Versión 2 (MLP con Dropout + Early Stopping) ===")
    red_v2 = construir_red_v2(input_dim=input_dim, dropout_rate=0.3)
    red_v2.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=obtener_callbacks(),
        verbose=1
    )
    prob_v2_val = red_v2.predict(X_val, verbose=0).flatten()
    pred_v2_val = (prob_v2_val > 0.5).astype(int)
    f1_v2 = f1_score(y_val, pred_v2_val)
    print(f"Red V2 -> F1-Score en Validación: {f1_v2:.4f}")
    red_v2.save("modelos/red_v2.keras")

    print("\n================ RESUMEN DE COMPARACIÓN ================")
    print(f"F1-Score Validación - Línea Base : {f1_base:.4f}")
    print(f"F1-Score Validación - Red V1      : {f1_v1:.4f}")
    print(f"F1-Score Validación - Red V2      : {f1_v2:.4f}")
    print("=========================================================")


if __name__ == "__main__":
    ejecutar_entrenamiento()
    