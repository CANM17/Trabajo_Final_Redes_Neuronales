"""
API REST construida con FastAPI para servir dos modelos de
clasificación de tráfico de red entrenados sobre el dataset NSL-KDD:

  • POST /predict/binary     → Binario: Normal vs Ataque
  • POST /predict/multiclass → Multiclase: Normal / DoS / Probe

Arquitectura:
    Cliente (Gradio / curl) → FastAPI → scaler.joblib → modelo.keras → JSON
"""

import os
import logging
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Rutas a los artefactos del modelo
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BIN_PATH   = os.path.join(BASE_DIR, "models", "modelo_base.keras")
MODEL_MULTI_PATH = os.path.join(BASE_DIR, "models", "modelo_transfer.keras")
SCALER_PATH      = os.path.join(BASE_DIR, "models", "scaler.joblib")

# Etiquetas de las 3 categorías del modelo multiclase
CLASS_LABELS = {0: "Normal", 1: "DoS", 2: "Probe"}

# Descripciones amigables de cada categoría
CLASS_DESCRIPTIONS = {
    "Normal" : "Tráfico de red legítimo, sin indicios de actividad maliciosa.",
    "DoS"    : "Ataque de Denegación de Servicio — intento de saturar o inhabilitar un recurso.",
    "Probe"  : "Escaneo de red — reconocimiento para identificar servicios vulnerables.",
}

# Diccionario global de recursos (cargados una sola vez al iniciar)
resources: dict = {}


# Ciclo de vida de la aplicación: carga de modelos al arrancar
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelos y scaler en memoria al iniciar el servidor."""
    logger.info("Cargando modelos y artefactos...")
    try:
        resources["scaler"]        = joblib.load(SCALER_PATH)
        resources["model_binary"]  = tf.keras.models.load_model(MODEL_BIN_PATH)
        resources["model_multi"]   = tf.keras.models.load_model(MODEL_MULTI_PATH)
        logger.info("✅ Modelos cargados correctamente.")
    except Exception as e:
        logger.error(f"Error al cargar modelos: {e}")
        raise RuntimeError(f"No se pudieron cargar los modelos: {e}")
    yield
    resources.clear()


# Instancia de la aplicación
app = FastAPI(
    title       = "IDS API — Detección de Intrusos en Red",
    description = "API REST para clasificar tráfico de red usando redes neuronales entrenadas sobre NSL-KDD.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Habilitar CORS para que Gradio (puerto 7860) pueda consultar la API (puerto 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Esquema de entrada — validado con Pydantic
class NetworkFeatures(BaseModel):
    """
    41 features del dataset NSL-KDD.
    El campo `features` debe ser una lista de exactamente 41 valores numéricos,
    correspondientes a las columnas del dataset en el orden oficial.
    """
    features: list[float] = Field(
        ...,
        description="Lista de 41 features del registro de red (orden NSL-KDD).",
        example=[0, 1, 0, 10, 491, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 150, 25, 0.17, 0.03,
                 0.17, 0, 0.01, 0.06, 0, 0],
    )

    @field_validator("features")
    @classmethod
    def validate_length(cls, v):
        if len(v) != 41:
            raise ValueError(f"Se esperan exactamente 41 features, se recibieron {len(v)}.")
        return v


# Función auxiliar de predicción
def _predict(features: list[float], mode: str) -> dict:
    """
    Ejecuta el pipeline completo: escalar → predecir → formatear respuesta.

    Parámetros:
    -----------
    features : list[float] — 41 valores de la conexión de red
    mode     : 'binary' | 'multiclass'

    Retorna:
    --------
    dict con prediction, probability/probabilities, description y status
    """
    try:
        # 1. Escalar (mismo transform que durante el entrenamiento)
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        X_scaled = resources["scaler"].transform(X)

        if mode == "binary":
            model = resources["model_binary"]
            prob  = float(model.predict(X_scaled, verbose=0)[0][0])
            THRESHOLD = 0.20
            label = "ATAQUE DETECTADO" if prob > THRESHOLD else "Tráfico Normal"
            return {
                "prediction"  : label,
                "probability" : round(prob, 4),
                "confidence"  : f"{prob * 100:.1f}%" if prob > THRESHOLD else f"{(1 - prob) * 100:.1f}%",
                "mode"        : "binary",
                "status"      : "success",
            }

        elif mode == "multiclass":
            model = resources["model_multi"]
            probs = model.predict(X_scaled, verbose=0)[0]  # shape (3,)
            idx   = int(np.argmax(probs))
            label = CLASS_LABELS[idx]
            return {
                "prediction"   : label,
                "description"  : CLASS_DESCRIPTIONS[label],
                "confidence"   : f"{float(probs[idx]) * 100:.1f}%",
                "probabilities": {CLASS_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
                "mode"         : "multiclass",
                "status"       : "success",
            }

    except Exception as e:
        logger.error(f"Error en predicción ({mode}): {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# Endpoints
@app.get("/", tags=["Info"])
def root():
    """Endpoint de bienvenida / health check."""
    return {
        "message"   : "IDS API — Sistema de Detección de Intrusos",
        "autor"     : "César Núñez",
        "version"   : "1.0.0",
        "endpoints" : ["/predict/binary", "/predict/multiclass", "/docs"],
    }


@app.get("/health", tags=["Info"])
def health():
    """Verifica que los modelos estén cargados en memoria."""
    loaded = all(k in resources for k in ("scaler", "model_binary", "model_multi"))
    return {"status": "ok" if loaded else "error", "models_loaded": loaded}


@app.post("/predict/binary", tags=["Predicción"])
def predict_binary(data: NetworkFeatures):
    """
    **Clasificación Binaria**

    Determina si el tráfico de red es **Normal** o un **Ataque**.

    - Input: 41 features del registro de red.
    - Output: etiqueta, probabilidad y confianza.
    """
    return _predict(data.features, mode="binary")


@app.post("/predict/multiclass", tags=["Predicción"])
def predict_multiclass(data: NetworkFeatures):
    """
    **Clasificación Multiclase** (Transfer Learning)

    Clasifica el tipo de tráfico en 3 categorías: **Normal / DoS / Probe**

    - Input: 41 features del registro de red.
    - Output: etiqueta, descripción, confianza y probabilidades por clase.
    """
    return _predict(data.features, mode="multiclass")
