Aquí tienes el contenido completo y corregido para tu archivo README.md. Puedes copiar todo el bloque de texto de abajo y pegarlo directamente en tu archivo:

Markdown
# IDS con Redes Neuronales — NSL-KDD

**Proyecto Final | Materia: Redes Neuronales | Estudiante: César Núñez**

Sistema de Detección de Intrusos (IDS) que clasifica tráfico de red usando dos modelos:
- **MLP Base**: Clasificación binaria (Normal vs Ataque).
- **Transfer Learning**: Clasificación multiclase en 5 categorías (Normal / DoS / Probe / R2L / U2R).

---

## 📁 Estructura del proyecto

proyecto_ids/
│
├── app/
│   └── main.py               # API REST con FastAPI (Puerto 8080)
│
├── models/                   # Artefactos del modelo
│   ├── modelo_base.keras
│   ├── modelo_transfer.keras
│   ├── scaler.joblib
│   └── encoders.joblib
│
├── venv/                     # Entorno virtual de Python (Local)
├── gradio_app.py             # Interfaz visual Gradio (Gradio 6.x)
├── requirements.txt          # Dependencias unificadas
└── .gitignore


---

## 🚀 Cómo ejecutar (Local VENV)

Debido a optimizaciones de espacio en disco, se recomienda la ejecución mediante entorno virtual para reducir el consumo de ~5 GB (Docker) a ~1.2 GB (Local).

### 1. Preparación del entorno
```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows
.\venv\Scripts\activate

# Instalar dependencias actualizadas
