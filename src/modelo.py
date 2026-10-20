"""
Módulo de Modelado para Detección de Spam (Spambase)
Integrante 2 - Etapa 3: Modelado y Comparación
"""

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------
# 1. Línea Base (Modelo Simple / Regresión Logística)
# ---------------------------------------------------------
def construir_linea_base():
    """Construye una Regresión Logística como línea base simple."""
    return LogisticRegression(max_iter=1000, random_state=42)


# ---------------------------------------------------------
# 2. Red Neuronal Versión 1 (MLP Básico)
# ---------------------------------------------------------
def construir_red_v1(input_dim=57):
    """
    Perceptrón Multicapa (MLP) básico:
    - Capa de entrada (57 características estadísticas de Spambase)
    - 2 capas ocultas con activación ReLU (64 y 32 unidades)
    - Capa de salida con activación Sigmoide (clasificación binaria)
    """
    modelo = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return modelo


# ---------------------------------------------------------
# 3. Red Neuronal Versión 2 (MLP con Dropout)
# ---------------------------------------------------------
def construir_red_v2(input_dim=57, dropout_rate=0.3):
    """
    MLP con regularización mediante Dropout:
    - Capas ocultas más profundas (128, 64 y 32 unidades)
    - Capas de Dropout (0.3) para prevenir el sobreajuste (overfitting)
    """
    modelo = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return modelo


# ---------------------------------------------------------
# 4. Callback de Parada Temprana (Early Stopping)
# ---------------------------------------------------------
def obtener_callbacks():
    """
    Callback para monitorear 'val_loss'. Si no mejora durante 10 épocas,
    detiene el entrenamiento y restaura los mejores pesos.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    return [early_stopping]