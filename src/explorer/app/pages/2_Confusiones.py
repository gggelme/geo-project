import numpy as np
import plotly.express as px
import streamlit as st

import comun
from explorer.core.metricas import matriz_confusion, pares_confusion

st.set_page_config(page_title="Confusiones", layout="wide")
ctx = comun.contexto()
df, config = ctx["df"], ctx["config"]

st.title("Matriz de confusión")
c = st.columns(4)
nivel = c[0].radio("Nivel", ["pais", "continente"], horizontal=True)
norm = c[1].radio("Normalizar", ["por fila (real)", "por columna (predicho)", "conteos"])
ocultar_diag = c[2].checkbox("Ocultar diagonal", value=True)
continentes = sorted(set(config["pais_a_continente"].values()))
filtro_cont = c[3].multiselect("Continentes (solo nivel país)", continentes, default=continentes)

cm = matriz_confusion(df, config, nivel)
if nivel == "pais":
    keep = [p for p in cm.index if config["pais_a_continente"][p] in filtro_cont]
    cm = cm.loc[keep, keep]
valores = cm.to_numpy().astype(float)
if norm.startswith("por fila"):
    valores = np.divide(valores, valores.sum(1, keepdims=True), out=np.zeros_like(valores),
                        where=valores.sum(1, keepdims=True) > 0)
elif norm.startswith("por columna"):
    valores = np.divide(valores, valores.sum(0, keepdims=True), out=np.zeros_like(valores),
                        where=valores.sum(0, keepdims=True) > 0)
if ocultar_diag:
    np.fill_diagonal(valores, 0)
fig = px.imshow(valores, x=list(cm.columns), y=list(cm.index), color_continuous_scale="Blues",
                labels={"x": "Predicho", "y": "Real", "color": "valor"}, aspect="auto",
                height=750 if nivel == "pais" else 450)
fig.update_xaxes(side="top", tickangle=90, dtick=1)
fig.update_yaxes(dtick=1)
st.plotly_chart(fig, width="stretch")

st.subheader("Pares más confundidos (real → predicho)")
m = st.columns(3)
min_n = m[0].number_input("Mínimo de errores", 1, 1000, 5)
solo = m[1].selectbox("Continente", ["Todos", "Mismo continente", "Distinto continente"])
orden = m[2].selectbox("Ordenar por", ["n_errores", "pct_de_real"])
pares = pares_confusion(df, config, int(min_n))
pares["pct_de_real"] *= 100
if solo == "Mismo continente":
    pares = pares[pares["mismo_continente"]]
elif solo == "Distinto continente":
    pares = pares[~pares["mismo_continente"]]
st.dataframe(pares.sort_values(orden, ascending=False), width="stretch", height=400,
             column_config={"pct_de_real": st.column_config.ProgressColumn(
                 "pct_de_real", min_value=0, max_value=100, format="%.1f%%")})
st.caption("Para ver las imágenes de un par, filtrá real y predicho en la página Galeria.")
