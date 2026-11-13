import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import comun
import mapas
from explorer.core.metricas import matriz_confusion, pares_confusion

ctx = comun.ctx()
df, config, clases = ctx["df"], ctx["config"], ctx["clases"]
es_cluster = ctx["tarea"] == "cluster"
nombre = ctx["nombre_clase"]
if es_cluster:
    clusters = ctx["clusters"]
    continente_de = dict(zip(clusters["etiqueta"], clusters["continente"]))
else:
    continente_de = config["pais_a_continente"]
continentes = sorted(set(continente_de.values()))

st.title("Matriz de confusión")
c = st.columns(4)
nivel = c[0].radio("Nivel", [nombre, "continente"], horizontal=True)
norm = c[1].radio("Normalizar", ["por fila (real)", "por columna (predicho)", "conteos"])
ocultar_diag = c[2].checkbox("Ocultar diagonal", value=True)
filtro_cont = c[3].multiselect(f"Continentes (solo nivel {nombre})", continentes, default=continentes)

if nivel == nombre:
    etiquetas = [p for p in clases if continente_de.get(p) in filtro_cont]
    cm = matriz_confusion(df, etiquetas, "clase")
else:
    cm = matriz_confusion(df, sorted(set(df["continente_real"].dropna()) | set(continentes)), "continente")
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
                height=750 if nivel == nombre else 450)
fig.update_xaxes(side="top", tickangle=90, dtick=1)
fig.update_yaxes(dtick=1)
st.plotly_chart(fig, width="stretch")

st.subheader("Pares más confundidos (real → predicho)")
m = st.columns(3)
min_n = m[0].number_input("Mínimo de errores", 1, 1000, 5)
solo = m[1].selectbox("Continente", ["Todos", "Mismo continente", "Distinto continente"])
orden = m[2].selectbox("Ordenar por", ["n_errores", "pct_de_real"] + (["distancia_km"] if es_cluster else []))
pares = pares_confusion(df, int(min_n), ctx.get("clusters"))
pares["pct_de_real"] *= 100
if solo == "Mismo continente":
    pares = pares[pares["mismo_continente"]]
elif solo == "Distinto continente":
    pares = pares[~pares["mismo_continente"]]
pares = pares.sort_values(orden, ascending=False)
st.dataframe(pares, width="stretch", height=400, hide_index=True,
             column_config={"pct_de_real": st.column_config.ProgressColumn(
                 "pct_de_real", min_value=0, max_value=100, format="%.1f%%"),
                 "distancia_km": st.column_config.NumberColumn("distancia entre prototipos km", format="%.0f")})

if es_cluster and len(pares):
    st.markdown("**Mapa de los pares** (línea real → predicho, grosor según errores; hasta 60 pares)")
    resumen = clusters.set_index("etiqueta")
    fig = mapas.figura(alto=520)
    mapas.bordes(fig, ctx["aristas"], color="#999", ancho=0.6)
    top = pares.head(60)
    maximo = top["n_errores"].max()
    for _, f in top.iterrows():
        a, b = resumen.loc[f["clase_real"]], resumen.loc[f["clase_predicha"]]
        la, lo = mapas.geodesica(a["lat"], a["lon"], b["lat"], b["lon"])
        fig.add_trace(go.Scattergeo(
            lat=la, lon=lo, mode="lines", showlegend=False, opacity=0.6, hoverinfo="skip",
            line=dict(width=1 + 6 * f["n_errores"] / maximo,
                      color="#1f77b4" if f["mismo_continente"] else "#d62728")))
    usados = resumen.loc[sorted(set(top["clase_real"]) | set(top["clase_predicha"]))]
    fig.add_trace(go.Scattergeo(
        lat=usados["lat"], lon=usados["lon"], mode="markers+text", text=[e.split(" ")[0] for e in usados.index],
        textposition="top center", textfont=dict(size=10), marker=dict(size=8, color="#333"),
        hovertext=list(usados.index), hoverinfo="text", showlegend=False))
    st.plotly_chart(fig, width="stretch")
    st.caption("Azul: mismo continente · rojo: distinto continente.")
st.caption("Para ver las imágenes de un par, filtrá real y predicho en la página Galeria.")
