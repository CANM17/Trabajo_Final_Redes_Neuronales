# ══════════════════════════════════════════════════════════════
#  Dockerfile — IDS API (FastAPI)
#  Imagen base: Python 3.10 slim para minimizar el tamaño final
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

# Metadatos del contenedor
LABEL maintainer="César Núñez"
LABEL description="API FastAPI para detección de intrusos con redes neuronales (NSL-KDD)"

# ── Variables de entorno ──────────────────────────────────────
# Evita que Python escriba archivos .pyc en disco
ENV PYTHONDONTWRITEBYTECODE=1
# Desactiva el buffering de stdout/stderr (logs en tiempo real)
ENV PYTHONUNBUFFERED=1

# ── Directorio de trabajo dentro del contenedor ──────────────
WORKDIR /proyecto

# ── Dependencias del sistema necesarias para compilar paquetes ─
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Instalar dependencias de Python ──────────────────────────
# Se copia primero requirements.txt para aprovechar la caché de Docker:
# si el código cambia pero no las dependencias, no se reinstalan.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copiar el código del proyecto ────────────────────────────
COPY . .

# ── Exponer el puerto de FastAPI ──────────────────────────────
EXPOSE 8080

# ── Comando de inicio ─────────────────────────────────────────
# --host 0.0.0.0 → acepta conexiones desde fuera del contenedor
# --port 8080    → puerto interno del contenedor
# --workers 1    → un worker (suficiente para demo académica)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
