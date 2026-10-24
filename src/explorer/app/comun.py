"""Barra lateral y carga compartida entre las páginas de la app."""
import sys
from pathlib import Path

# Permite `import explorer...` al correr con `streamlit run src/explorer/app/Inicio.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import torch

from explorer.core.paths import listar_corridas, listar_datasets
from explorer.core.datos import cargar_metadatos
from explorer.core.inferencia import predecir
from explorer.core.metricas import tabla_predicciones
from explorer.core.modelo import cargar_modelo


@st.cache_resource(show_spinner="Cargando modelo...")
def modelo_cacheado(ruta_pt, device):
    return cargar_modelo(ruta_pt, device)


@st.cache_data(show_spinner="Cargando metadatos...")
def metadatos_cacheados(ruta):
    return cargar_metadatos(ruta)


@st.cache_data(show_spinner=False)
def tabla_cacheada(ruta_pt, ruta_parquet, device):
    meta = metadatos_cacheados(ruta_parquet)
    res = predecir(ruta_pt, ruta_parquet, device)
    config = modelo_cacheado(ruta_pt, device)[1]
    return tabla_predicciones(meta, res, config), res


def contexto():
    """Dibuja los selectores y devuelve el contexto (o detiene la página si no hay datos)."""
    corridas, datasets = listar_corridas(), listar_datasets()
    if not corridas or not datasets:
        st.error("No hay corridas en outputs/ o parquets en datasets/.")
        st.stop()
    with st.sidebar:
        st.header("Selección")
        corrida = st.selectbox("Corrida (pesos)", corridas, format_func=lambda c: c["etiqueta"],
                               key="sel_corrida")
        dataset = st.selectbox("Dataset", datasets, format_func=lambda d: d["etiqueta"],
                               key="sel_dataset")
        opciones = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox("Dispositivo", opciones, key="sel_device")
    ruta_pt, ruta_parquet = str(corrida["checkpoint"]), str(dataset["ruta"])
    with st.spinner("Corriendo el modelo sobre el dataset (solo la primera vez)..."):
        df, res = tabla_cacheada(ruta_pt, ruta_parquet, device)
    modelo, config, transform = modelo_cacheado(ruta_pt, device)
    if not df["en_modelo"].all():
        fuera = sorted(df.loc[~df["en_modelo"], "country_code"].unique())
        st.sidebar.warning(f"{(~df['en_modelo']).sum()} imágenes de países fuera del modelo "
                           f"({', '.join(fuera)}) se excluyen de las métricas.")
    return {"df": df, "res": res, "modelo": modelo, "config": config, "transform": transform,
            "corrida": corrida, "dataset": dataset, "ruta_pt": ruta_pt,
            "ruta_parquet": ruta_parquet, "device": device}
