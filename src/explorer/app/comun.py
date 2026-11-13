"""Barra lateral y carga compartida entre las páginas de la app."""
import sys
from pathlib import Path

# Permite `import explorer...` al correr con `streamlit run src/explorer/app/Inicio.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import torch

from explorer.core.clusters import cargar_prototipos, contornos_voronoi, aristas_voronoi, describir_clusters
from explorer.core.paths import listar_corridas, listar_datasets, resolver_ruta, raiz_proyecto
from explorer.core.datos import cargar_metadatos
from explorer.core.inferencia import predecir
from explorer.core.metricas import tabla_predicciones
from explorer.core.modelo import cargar_modelo, tarea

NOMBRE_CLASE = {"pais": "país", "cluster": "cluster"}

# Los nombres de corridas y datasets son largos: el valor elegido se parte en varias líneas
# y la lista desplegable se ensancha hasta el texto completo (Streamlit no da tooltip por opción).
_CSS_SELECTORES = """<style>
section[data-testid="stSidebar"] div[data-baseweb="select"] div {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
div[data-baseweb="popover"] > div { width: max-content !important; max-width: min(90vw, 900px) !important; }
div[data-baseweb="popover"] li, div[data-baseweb="popover"] [role="option"] { white-space: nowrap !important; }
</style>"""


@st.cache_resource(show_spinner="Cargando modelo...")
def modelo_cacheado(ruta_pt, device):
    return cargar_modelo(ruta_pt, device)


@st.cache_data(show_spinner="Cargando metadatos...")
def metadatos_cacheados(ruta):
    return cargar_metadatos(ruta)


@st.cache_data(show_spinner="Describiendo clusters...")
def clusters_cacheados(ruta_pt, ruta_parquet, device):
    """(descripción por cluster, aviso, conjunto usado para nombrarlos)."""
    config = modelo_cacheado(ruta_pt, device)[1]
    protos, aviso = cargar_prototipos(config, Path(ruta_pt).parent)
    ruta_ref = resolver_ruta(config.get("ruta_train"))
    if ruta_ref is None:  # sin el train, se nombran con el dataset evaluado
        ruta_ref = Path(ruta_parquet)
    desc = describir_clusters(protos, metadatos_cacheados(str(ruta_ref)), config["pais_a_continente"])
    try:
        fuente = str(ruta_ref.relative_to(raiz_proyecto()))
    except ValueError:
        fuente = str(ruta_ref)
    return desc, aviso, fuente


@st.cache_data(show_spinner=False)
def geometria_cacheada(coords):
    """Bordes de Voronoi y contorno de cada celda, para los mapas."""
    return aristas_voronoi(coords), contornos_voronoi(coords)


@st.cache_data(show_spinner=False)
def tabla_cacheada(ruta_pt, ruta_parquet, device):
    meta = metadatos_cacheados(ruta_parquet)
    res = predecir(ruta_pt, ruta_parquet, device)
    config = modelo_cacheado(ruta_pt, device)[1]
    clusters = clusters_cacheados(ruta_pt, ruta_parquet, device)[0] if tarea(config) == "cluster" else None
    return tabla_predicciones(meta, res, config, clusters), res


def _dataset_de_validacion(corrida, datasets):
    ruta = resolver_ruta(corrida.get("ruta_validation"))
    return next((d for d in datasets if ruta is not None and d["ruta"].resolve() == ruta.resolve()), None)


def contexto():
    """Dibuja los selectores y devuelve el contexto (o detiene la página si no hay datos)."""
    corridas, datasets = listar_corridas(), listar_datasets()
    if not corridas or not datasets:
        st.error("No hay corridas en outputs/ o parquets en datasets/.")
        st.stop()
    with st.sidebar:
        st.markdown(_CSS_SELECTORES, unsafe_allow_html=True)
        st.header("Selección")
        corrida = st.selectbox("Corrida (pesos)", corridas, format_func=lambda c: c["etiqueta"],
                               key="sel_corrida")
        st.caption(f"**{corrida['carpeta'].parent.name}** / {corrida['carpeta'].name}  \n"
                   + " · ".join(corrida["etiqueta"].split(" · ")[1:]))
        # Al cambiar de corrida se preselecciona su dataset de validación.
        if st.session_state.get("_corrida_previa") != str(corrida["carpeta"]):
            st.session_state["_corrida_previa"] = str(corrida["carpeta"])
            propio = _dataset_de_validacion(corrida, datasets)
            if propio is not None:
                st.session_state["sel_dataset"] = propio
        dataset = st.selectbox("Dataset", datasets, format_func=lambda d: d["etiqueta"],
                               key="sel_dataset")
        opciones = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox("Dispositivo", opciones, key="sel_device")
    ruta_pt, ruta_parquet = str(corrida["checkpoint"]), str(dataset["ruta"])
    modelo, config, transform = modelo_cacheado(ruta_pt, device)
    modo = tarea(config)
    with st.spinner("Corriendo el modelo sobre el dataset (solo la primera vez)..."):
        df, res = tabla_cacheada(ruta_pt, ruta_parquet, device)
    ctx = {"df": df, "res": res, "modelo": modelo, "config": config, "transform": transform,
           "corrida": corrida, "dataset": dataset, "ruta_pt": ruta_pt, "ruta_parquet": ruta_parquet,
           "device": device, "tarea": modo, "nombre_clase": NOMBRE_CLASE[modo]}
    if modo == "cluster":
        clusters, aviso, fuente = clusters_cacheados(ruta_pt, ruta_parquet, device)
        aristas, contornos = geometria_cacheada(clusters[["lat", "lon"]].to_numpy(float))
        ctx |= {"clusters": clusters, "clases": list(clusters["etiqueta"]), "fuente_clusters": fuente,
                "aristas": aristas, "contornos": contornos}
        if aviso:
            st.sidebar.warning(aviso)
        if not df["en_modelo"].all():
            st.sidebar.warning(f"{(~df['en_modelo']).sum()} imágenes sin coordenadas se excluyen.")
    else:
        ctx["clases"] = list(config["paises"])
        if not df["en_modelo"].all():
            fuera = sorted(df.loc[~df["en_modelo"], "country_code"].fillna("(vacío)").astype(str).unique())
            st.sidebar.warning(f"{(~df['en_modelo']).sum()} imágenes de países fuera del modelo "
                               f"({', '.join(fuera)}) se excluyen de las métricas.")
    st.session_state["_ctx"] = ctx
    return ctx


def ctx():
    """Contexto armado por el entrypoint (Inicio.py) en esta misma ejecución."""
    if "_ctx" not in st.session_state:
        st.error("Abrí la app desde Inicio.py: `streamlit run src/explorer/app/Inicio.py`.")
        st.stop()
    return st.session_state["_ctx"]
