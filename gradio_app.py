"""
IDS — Gradio Local (Versión Unificada con HF)
Mismo motor de predicción que Hugging Face para resultados idénticos.
"""

import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- CARGA DIRECTA DE MODELOS (Sin pasar por API) ---
try:
    model_bin = tf.keras.models.load_model(os.path.join(MODELS_DIR, "modelo_base.keras"))
    model_multi = tf.keras.models.load_model(os.path.join(MODELS_DIR, "modelo_transfer.keras"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
    print("✅ Modelos y artefactos cargados en local. Resultados sincronizados con HF.")
except Exception as e:
    print(f"❌ Error al cargar modelos: {e}")

# Mapeo multiclase idéntico a HF
MULTI_CLASSES = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}

def analyze_file(file):
    if file is None: return None, "Cargue un archivo CSV."
    
    try:
        # 1. Carga de datos (Ignora columnas extra de etiquetas/dificultad)
        df_raw = pd.read_csv(file.name, header=None).iloc[:, :41]
        df_proc = df_raw.copy()

        # 2. Preprocesamiento Vectorial (Batch)
        if encoders:
            for col, pos in [('protocol_type', 1), ('service', 2), ('flag', 3)]:
                df_proc[pos] = encoders[col].transform(df_proc[pos].astype(str))

        # Convertir a float y escalar
        X = df_proc.values.astype(float)
        if scaler:
            X = scaler.transform(X)

        # 3. Predicciones directas en memoria
        preds_bin = model_bin.predict(X, verbose=0)
        preds_multi = model_multi.predict(X, verbose=0)

        results = []
        ataques_count = 0

        for i in range(len(X)):
            # Lógica Binaria (Umbral 0.5 estricto)
            prob_bin = float(preds_bin[i][0])
            is_attack = prob_bin > 0.5
            
            label_bin = "🛑 Ataque" if is_attack else "✅ Normal"
            conf_bin = f"{prob_bin:.2%}" if is_attack else f"{(1-prob_bin):.2%}"
            
            if is_attack:
                ataques_count += 1

            # Lógica Multiclase
            idx_multi = np.argmax(preds_multi[i])
            label_multi = MULTI_CLASSES.get(idx_multi, "Otros")
            conf_multi = f"{np.max(preds_multi[i]):.2%}"

            results.append([i+1, label_bin, conf_bin, label_multi, conf_multi])

        # 4. Resumen final idéntico
        total = len(results)
        resumen = (f"📊 Análisis Completado (Motor Local)\n"
                   f"-------------------\n"
                   f"Total registros: {total}\n"
                   f"✅ Tráfico Normal: {total - ataques_count}\n"
                   f"🛑 Ataques: {ataques_count}")
        
        return results, resumen

    except Exception as e:
        return None, f"Error en el procesamiento local: {str(e)}"

# --- INTERFAZ ---
with gr.Blocks(theme=gr.themes.Soft(), title="IDS Local Unificado") as demo:
    gr.Markdown("# 🛡️ IDS Local (Resultados Sincronizados)")
    gr.Markdown("Esta versión utiliza los modelos cargados directamente en memoria para garantizar consistencia con la versión de la nube.")
    
    with gr.Row():
        file_input = gr.File(label="Archivo CSV (NSL-KDD)", file_types=[".csv"])
        btn = gr.Button("🚀 Ejecutar Análisis", variant="primary")
    
    with gr.Row():
        summary_out = gr.Textbox(label="Estado del Análisis", lines=6)
        table_out = gr.DataFrame(
            headers=["#", "Clasificación", "Confianza", "Tipo", "Conf. Tipo"],
            label="Resultados por registro"
        )
    
    btn.click(fn=analyze_file, inputs=file_input, outputs=[table_out, summary_out])

if __name__ == "__main__":
    # Ya no necesitas tener corriendo el servidor Uvicorn para que esto funcione
    demo.launch(server_port=7860)