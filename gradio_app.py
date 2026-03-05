"""
IDS — Interfaz Gradio
Estudiante: César Núñez | Materia: Redes Neuronales
Comunica con la API FastAPI via HTTP (no carga modelos directamente).
"""

import gradio as gr
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://localhost:8080")

FEATURE_COLS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
]

DEFAULT_NORMAL = [
    0, 2, 30, 10, 491, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 2, 2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    150, 25, 0.17, 0.03, 0.17, 0.0, 0.01, 0.06, 0.0, 0.0,
]

PROFILE_DOS = [
    0, 2, 30, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 511, 511, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    255, 255, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
]

PROFILE_PROBE = [
    0, 2, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    255, 10, 0.04, 0.97, 0.04, 0.97, 0.0, 0.0, 0.0, 0.0,
]


def call_api(features, endpoint):
    try:
        resp = requests.post(
            f"{API_URL}/{endpoint}",
            json={"features": features},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"No se puede conectar a la API en {API_URL}"}
    except Exception as e:
        return {"error": str(e)}


def analyze_csv(file):
    if file is None:
        return None, "Cargue un archivo CSV para comenzar."
    try:
        df = pd.read_csv(file.name, header=None)
    except Exception as e:
        return None, f"Error al leer el archivo: {str(e)}"

    if df.shape[1] == 43:
        df = df.iloc[:, :41]
    elif df.shape[1] != 41:
        return None, f"El archivo tiene {df.shape[1]} columnas. Se esperan 41 o 43."

    df.columns = FEATURE_COLS
    resultados = []
    errores = 0

    for idx, row in df.iterrows():
        features = row.tolist()
        r_bin = call_api(features, "predict/binary")
        if "error" in r_bin:
            errores += 1
            resultados.append({"#": idx+1, "Binario": "ERROR", "Conf. Binario": "-", "Tipo Ataque": "ERROR", "Conf. Tipo": "-"})
            continue

        pred_bin  = r_bin.get("prediction", "-")
        prob_bin  = r_bin.get("probability", 0)
        conf_bin  = f"{prob_bin:.1%}"

        r_multi = call_api(features, "predict/multiclass")
        if "error" in r_multi:
            pred_multi, conf_multi = "ERROR", "-"
        else:
            pred_multi = r_multi.get("prediction", "-")
            probs      = r_multi.get("probabilities", {})
            conf_multi = f"{max(probs.values()):.1%}" if probs else "-"

        resultados.append({"#": idx+1, "Binario": pred_bin, "Conf. Binario": conf_bin, "Tipo Ataque": pred_multi, "Conf. Tipo": conf_multi})

    resultado_df = pd.DataFrame(resultados)
    total    = len(resultado_df)
    ataques  = resultado_df["Binario"].str.contains("ATAQUE", case=False, na=False).sum()
    normales = total - ataques - errores

    resumen = (
        f"Conexiones analizadas : {total}\n"
        f"Trafico normal        : {normales} ({normales/total:.1%})\n"
        f"Ataques detectados    : {ataques} ({ataques/total:.1%})\n"
    )
    if errores:
        resumen += f"Errores de API        : {errores}\n"

    return resultado_df, resumen


def predict_manual(duration, src_bytes, dst_bytes, count, srv_count,
                   serror_rate, same_srv_rate, diff_srv_rate, profile):
    if profile == "Simular ataque DoS":
        features = PROFILE_DOS.copy()
    elif profile == "Simular escaneo (Probe)":
        features = PROFILE_PROBE.copy()
    else:
        features = DEFAULT_NORMAL.copy()

    features[0]  = float(duration)
    features[4]  = float(src_bytes)
    features[5]  = float(dst_bytes)
    features[22] = float(count)
    features[23] = float(srv_count)
    features[24] = float(serror_rate)
    features[28] = float(same_srv_rate)
    features[29] = float(diff_srv_rate)

    r_bin = call_api(features, "predict/binary")
    if "error" in r_bin:
        return r_bin["error"], ""

    pred_bin  = r_bin.get("prediction", "-")
    prob_bin  = r_bin.get("probability", 0)
    conf_bin  = r_bin.get("confidence", "-")
    is_attack = prob_bin > 0.20
    bar       = "█" * int(prob_bin * 20) + "░" * (20 - int(prob_bin * 20))

    r_multi = call_api(features, "predict/multiclass")
    tipo = r_multi.get("prediction", "-") if "error" not in r_multi else "-"

    status = f"""{'[ALERTA] INTRUSION DETECTADA' if is_attack else '[OK] TRAFICO LEGITIMO'}
{'━' * 40}
Resultado binario : {pred_bin}
Tipo de ataque    : {tipo}
Probabilidad      : {bar}  {prob_bin:.1%}
Confianza         : {conf_bin}
{'━' * 40}
{'Recomendacion: bloquear esta conexion.' if is_attack else 'Conexion dentro de parametros normales.'}"""

    return status, f"Probabilidad de ataque: {prob_bin:.4f} | Umbral: 0.20"


THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")

with gr.Blocks(theme=THEME, title="IDS — Deteccion de Intrusos") as demo:

    gr.Markdown("# Sistema de Deteccion de Intrusos con Redes Neuronales\n**Proyecto Final — Redes Neuronales | Cesar Nuñez**")

    with gr.Tabs():

        with gr.Tab("Analisis por Archivo"):
            gr.Markdown("### Cargue un archivo CSV con registros de red\nEl archivo debe tener **41 columnas** sin encabezados (o 43 con label y difficulty al final).")
            with gr.Row():
                with gr.Column(scale=1):
                    file_input  = gr.File(label="Archivo CSV", file_types=[".csv"])
                    btn_csv     = gr.Button("Analizar archivo", variant="primary", size="lg")
                    out_resumen = gr.Textbox(label="Resumen", lines=5, interactive=False, placeholder="El resumen aparecera aqui...")
                with gr.Column(scale=2):
                    out_tabla = gr.Dataframe(label="Resultados por conexion", interactive=False)
            btn_csv.click(fn=analyze_csv, inputs=[file_input], outputs=[out_tabla, out_resumen])

        with gr.Tab("Analisis Manual"):
            gr.Markdown("### Ingrese los parametros manualmente — ambos modelos en simultaneo")
            with gr.Row():
                with gr.Column(scale=1):
                    inputs_manual = [
                        gr.Slider(0, 60000, value=0,   label="Duration (seg)", step=1),
                        gr.Slider(0, 1e7,   value=491, label="Source Bytes", step=1),
                        gr.Slider(0, 1e7,   value=0,   label="Destination Bytes", step=1),
                        gr.Slider(0, 511,   value=2,   label="Count (conexiones al mismo host)", step=1),
                        gr.Slider(0, 511,   value=2,   label="Srv Count (mismo servicio)", step=1),
                        gr.Slider(0.0, 1.0, value=0.0, label="SYN Error Rate", step=0.01),
                        gr.Slider(0.0, 1.0, value=1.0, label="Same Service Rate", step=0.01),
                        gr.Slider(0.0, 1.0, value=0.0, label="Different Service Rate", step=0.01),
                        gr.Radio(choices=["Perfil neutro", "Simular ataque DoS", "Simular escaneo (Probe)"],
                                 value="Perfil neutro", label="Perfil de carga rapida"),
                    ]
                    btn_manual = gr.Button("Analizar", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out_status = gr.Textbox(label="Veredicto", lines=9, interactive=False)
                    out_detail = gr.Textbox(label="Detalle tecnico", lines=2, interactive=False)
            btn_manual.click(fn=predict_manual, inputs=inputs_manual, outputs=[out_status, out_detail])
            gr.Examples(
                examples=[
                    [0, 491, 0, 2, 2, 0.0, 1.0, 0.0, "Perfil neutro"],
                    [0, 0, 0, 511, 511, 1.0, 1.0, 0.0, "Simular ataque DoS"],
                    [0, 0, 0, 1, 1, 0.0, 1.0, 1.0, "Simular escaneo (Probe)"],
                ],
                inputs=inputs_manual,
                label="Ejemplos de prueba",
            )

        with gr.Tab("Acerca del proyecto"):
            gr.Markdown("""
## IDS con Redes Neuronales — NSL-KDD

### Dataset
El **NSL-KDD** es la version depurada del KDD Cup 1999. Contiene registros TCP/IP con **41 variables**.
Las categorias R2L y U2R fueron excluidas del modelo por desequilibrio severo (menos del 1% del dataset),
siguiendo la practica estandar en la literatura del NSL-KDD.

### Modelos
| Modelo | Tipo | Arquitectura |
|---|---|---|
| **MLP Base** | Binario (Normal vs Ataque) | Dense(64) → Dropout → Dense(32) → Sigmoid |
| **Transfer Learning** | Multiclase (3 categorias) | Base congelado + Dense(16) → Softmax(3) |

### Categorias del modelo multiclase
| Categoria | Descripcion |
|---|---|
| **Normal** | Trafico legitimo |
| **DoS** | Denegacion de Servicio |
| **Probe** | Escaneo de red |

### Stack tecnologico
```
TensorFlow / Keras  →  FastAPI  →  Gradio
```
### Autor
**Cesar Nuñez** — Proyecto Final, Materia: Redes Neuronales
            """)

    gr.Markdown("---\n*IDS con Redes Neuronales — Cesar Nuñez*")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
