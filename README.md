# IDS con Redes Neuronales — NSL-KDD

**Proyecto Final | Materia: Redes Neuronales | Estudiante: César Núñez**

Sistema de Detección de Intrusos (IDS) que clasifica tráfico de red usando dos modelos:
- **MLP Base**: Clasificación binaria (Normal vs Ataque) con umbral 0.20.
- **Transfer Learning**: Clasificación multiclase en 3 categorías (Normal / DoS / Probe).

> Las categorías R2L y U2R fueron excluidas por desequilibrio severo (<1% del dataset), práctica estándar en la literatura del NSL-KDD.

---

## 📁 Estructura del proyecto

```
Proyecto_RN/
├── app/
│   └── main.py               # API REST con FastAPI (Puerto 8080)
├── models/                   # Artefactos del modelo (generados por el notebook)
│   ├── modelo_base.keras
│   ├── modelo_transfer.keras
│   ├── scaler.joblib
│   └── encoders.joblib
├── data/
│   ├── train.csv
│   └── test.csv
├── notebook/
│   └── entrenamiento.ipynb   # Entrenamiento completo
├── gradio_app.py             # Interfaz visual Gradio
├── requirements.txt          # Dependencias API
├── requirements.gradio.txt   # Dependencias Gradio
├── Dockerfile.api
├── Dockerfile.gradio
└── docker-compose.yml
```

---

## 🐳 Cómo ejecutar con Docker (recomendado)

### Requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.

### Pasos

```bash
# 1. Asegurarse de tener los modelos entrenados en la carpeta models/
#    (correr primero el notebook entrenamiento.ipynb)

# 2. Desde la raíz del proyecto, construir y levantar ambos servicios
docker compose up --build

# Para correr en background
docker compose up --build -d

# Para detener
docker compose down
```

### URLs disponibles
| Servicio | URL |
|---|---|
| Interfaz Gradio | http://localhost:7860 |
| API FastAPI | http://localhost:8080 |
| Documentación API (Swagger) | http://localhost:8080/docs |

> Si reentrenás los modelos no es necesario reconstruir las imágenes — los modelos se montan como volumen. Solo reiniciá el servicio con `docker compose restart api`.

---

## 💻 Cómo ejecutar localmente (sin Docker)

```bash
# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements.gradio.txt

# Terminal 1 — levantar la API
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Terminal 2 — levantar Gradio
python gradio_app.py
```

---

## 🧠 Modelos

| Modelo | Accuracy | AUC-ROC | Notas |
|---|---|---|---|
| MLP Base (Binario) | 90% | 0.984 | Umbral 0.20 para maximizar recall de ataques |
| Transfer Learning (3 clases) | 86% | — | Normal F1=0.90, DoS F1=0.85, Probe F1=0.75 |

## Stack tecnológico

```
TensorFlow / Keras  →  FastAPI  →  Gradio
```
