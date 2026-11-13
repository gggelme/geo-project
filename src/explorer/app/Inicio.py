"""Entrypoint: `streamlit run src/explorer/app/Inicio.py`.

Dibuja la selección (barra lateral) y arma el menú según la tarea de la corrida:
por país (Países) o por cluster (Clusters). Las páginas leen el contexto con comun.ctx().
"""
import streamlit as st

import comun

st.set_page_config(page_title="Explorador de modelo", layout="wide")
ctx = comun.contexto()

if ctx["tarea"] == "cluster":
    clases = st.Page("vistas/1_Clusters.py", title="Clusters", icon=":material/hexagon:", url_path="clases")
else:
    clases = st.Page("vistas/1_Paises.py", title="Países", icon=":material/flag:", url_path="clases")

pagina = st.navigation([
    st.Page("vistas/0_Resumen.py", title="Inicio", icon=":material/home:", default=True),
    clases,
    st.Page("vistas/2_Confusiones.py", title="Confusiones", icon=":material/grid_on:"),
    st.Page("vistas/3_Galeria.py", title="Galería", icon=":material/photo_library:"),
])
pagina.run()
