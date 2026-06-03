import io

import pandas as pd
import streamlit as st
from PIL import Image

from predictor import DEFAULT_CONFIDENCE, load_model, run_prediction

st.set_page_config(
    page_title="TechVet — Detecção de Neutrófilos",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 TechVet - Detecção de Neutrófilos")
st.markdown(
    "Faça upload de uma imagem microscópica para detectar automaticamente **neutrófilos** "
    "utilizando um modelo YOLOv8 treinado com dados veterinários."
)
st.divider()


@st.cache_resource(show_spinner="Carregando modelo...")
def get_model():
    return load_model()


with st.sidebar:
    st.header("Configurações")
    confidence = st.slider(
        "Limiar de confiança",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help="Detecções abaixo desse valor são descartadas.",
    )
    st.divider()
    st.caption("Modelo: YOLOv8n fine-tuned")
    st.caption("Dataset: Roboflow techvet v2 (CC BY 4.0)")
    st.caption("mAP@50: 0.962 | F1: 0.901")

uploaded_file = st.file_uploader(
    "Selecione uma imagem (JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    model = get_model()
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analisando imagem..."):
        result = run_prediction(model, image, confidence=confidence)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagem original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Resultado da detecção")
        st.image(result["annotated_image"], use_container_width=True)

    st.divider()

    count = result["count"]
    detections = result["detections"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Neutrófilos detectados", count)

    if detections:
        avg_conf = sum(d["confidence"] for d in detections) / count
        max_conf = max(d["confidence"] for d in detections)
        metric_col2.metric("Confiança média", f"{avg_conf:.1%}")
        metric_col3.metric("Confiança máxima", f"{max_conf:.1%}")

    if detections:
        st.subheader("Detalhes das detecções")
        df = pd.DataFrame(
            [
                {
                    "#": i + 1,
                    "Classe": d["class_name"],
                    "Confiança": f"{d['confidence']:.1%}",
                    "x1": int(d["box"][0]),
                    "y1": int(d["box"][1]),
                    "x2": int(d["box"][2]),
                    "y2": int(d["box"][3]),
                }
                for i, d in enumerate(detections)
            ]
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

    annotated_bytes = io.BytesIO()
    result["annotated_image"].save(annotated_bytes, format="JPEG", quality=95)
    st.download_button(
        label="Baixar imagem anotada",
        data=annotated_bytes.getvalue(),
        file_name=f"tecvet_{uploaded_file.name}",
        mime="image/jpeg",
    )

    if count == 0:
        st.info("Nenhum neutrófilo detectado. Tente reduzir o limiar de confiança.")
else:
    st.info("Faça upload de uma imagem para começar a análise.")
