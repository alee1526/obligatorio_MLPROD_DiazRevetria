import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Clasificador de Lesiones de Piel", page_icon="🔬")
st.title("🔬 Clasificador de Lesiones de Piel")
st.caption("PAD-UFES-20 · demo multimodal (imagen + datos clínicos)")

st.info("Fase 0: andamiaje. La interfaz completa se construye en fases posteriores.")

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    if r.ok:
        st.success(f"API conectada en {API_URL} → {r.json()}")
    else:
        st.error(f"API respondió {r.status_code}")
except requests.RequestException as e:
    st.warning(f"No se pudo conectar a la API ({API_URL}): {e}")
