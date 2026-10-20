"""
Módulo src/api.py
-----------------
API de inferencia para el sistema de detección de spam (Spambase).
Carga el modelo promovido (modelo_produccion.joblib) y el preprocesador,
exponiendo endpoints de salud y predicción con validación de esquemas.

Autor: Grupo 6 - Tecnologías Emergentes
"""

from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="API de Detección de Spam - Spambase",
    version="1.0.0",
    description="API RESTful para clasificar correos electrónicos en Spam o No Spam usando el modelo promovido en E4."
)

# Configuración de rutas de artefactos
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / 'modelos'
DATOS_DIR = BASE_DIR / 'datos'

# Cargar artefactos en memoria al iniciar el servidor
modelo_path = MODELOS_DIR / 'modelo_produccion.joblib'
preprocesador_path = DATOS_DIR / 'preprocesador_referencia_train.joblib'

try:
    modelo = joblib.load(modelo_path)
    preprocesador = joblib.load(preprocesador_path)
    print("-> Modelo de producción y preprocesador cargados exitosamente en la API.")
except Exception as e:
    print(f"⚠️ Advertencia al cargar artefactos: {e}")
    modelo = None
    preprocesador = None

# Esquema de validación de entrada con Pydantic
# El dataset Spambase cuenta con 57 variables predictoras de frecuencias de palabras/caracteres.


class CorreoFeatures(BaseModel):
    features: list[float] = Field(
        ...,
        description="Lista exacta de 57 valores numéricos correspondientes a las características del dataset Spambase."
    )


@app.get("/salud", tags=["Monitoreo"])
def salud_api():
    """
    Endpoint de salud para verificar que la API está operativa y los modelos cargados.
    """
    if modelo is None or preprocesador is None:
        raise HTTPException(
            status_code=500, detail="Los artefactos del modelo no se encuentran cargados en el servidor.")
    return {
        "estado": "ok",
        "mensaje": "La API de detección de spam está funcionando correctamente",
        "modelo_activo": "modelo_produccion.joblib"
    }


@app.post("/predecir", tags=["Inferencia"])
def predecir_spam(datos: CorreoFeatures):
    """
    Recibe un vector de 57 características, aplica el preprocesamiento sin fuga 
    y retorna la predicción (Spam / No Spam) junto con su probabilidad.
    """
    if modelo is None or preprocesador is None:
        raise HTTPException(
            status_code=500, detail="Modelo no disponible en el servidor.")

    # Validar que la entrada tenga exactamente las 57 características del Spambase
    if len(datos.features) != 57:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban exactamente 57 características, pero se recibieron {len(datos.features)}."
        )

    try:
        # Convertir a arreglo 2D para el pipeline de scikit-learn
        X_entrada = np.array([datos.features])

        # Aplicar transformación (escalado/imputación guardada en E2)
        X_procesado = preprocesador.transform(X_entrada)

        # Predicción y probabilidad
        prediccion = int(modelo.predict(X_procesado)[0])

        # Obtener probabilidades si el modelo lo soporta
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(X_procesado)[0]
            probabilidad_spam = float(probabilidades[1])
        else:
            probabilidad_spam = 1.0 if prediccion == 1 else 0.0

        etiqueta_texto = "Spam" if prediccion == 1 else "No Spam"

        return {
            "prediccion_clase": prediccion,
            "etiqueta": etiqueta_texto,
            "probabilidad_spam": round(probabilidad_spam, 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error durante el procesamiento de la inferencia: {str(e)}")
