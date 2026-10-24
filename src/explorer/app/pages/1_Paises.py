import pandas as pd
import plotly.express as px
import streamlit as st

import comun
from explorer.core.metricas import resumen_por_pais

st.set_page_config(page_title="Países", layout="wide")
ctx = comun.contexto()
df, config = ctx["df"], ctx["config"]
resumen = resumen_por_pais(df, config)

st.title("Rendimiento por país")
continentes = sorted(resumen["continente"].unique())
sel_cont = st.multiselect("Continente", continentes, default=continentes)
vista = resumen[resumen["continente"].isin(sel_cont)]
st.dataframe(
    vista.sort_values("recall"),
    width="stretch", height=420,
    column_config={
        "recall": st.column_config.ProgressColumn("recall", min_value=0, max_value=1, format="%.2f"),
        "precision": st.column_config.NumberColumn(format="%.2f"),
        "f1": st.column_config.NumberColumn(format="%.2f"),
        "top5": st.column_config.NumberColumn(format="%.2f"),
        "conf_aciertos": st.column_config.NumberColumn(format="%.2f"),
        "conf_errores": st.column_config.NumberColumn(format="%.2f"),
    })

st.divider()
pais = st.selectbox("Detalle de país", list(vista.index))
d = df[df["en_modelo"]]


def barras(serie, titulo, color):
    serie = serie[serie > 0].sort_values(ascending=False).head(15)
    if serie.empty:
        st.write("Sin casos.")
        return
    fig = px.bar(pd.DataFrame({"pais": serie.index, "n": serie.values}), x="pais", y="n", title=titulo)
    fig.update_traces(marker_color=color)
    fig.update_layout(xaxis_title=None, yaxis_title="imágenes")
    st.plotly_chart(fig, width="stretch")


izq, der = st.columns(2)
with izq:
    reales = d[d["pais_real"] == pais]
    st.markdown(f"**Cuando es {pais}** ({len(reales)} imgs, recall {resumen.loc[pais, 'recall']:.1%}), "
                "el modelo predice:")
    barras(reales["pais_predicho"].value_counts(), f"Real = {pais} → predicho", "#4C78A8")
with der:
    predichas = d[d["pais_predicho"] == pais]
    st.markdown(f"**Cuando predice {pais}** ({len(predichas)} imgs, "
                f"precisión {resumen.loc[pais, 'precision']:.1%}), en realidad era:")
    barras(predichas["pais_real"].value_counts(), f"Predicho = {pais} ← real", "#E45756")

b1, b2, _ = st.columns([1, 1, 2])
if b1.button(f"Ver imágenes reales de {pais}"):
    st.session_state["filtro_galeria"] = {"real": [pais], "pred": [], "acierto": "Todos"}
    st.switch_page("pages/3_Galeria.py")
if b2.button(f"Ver imágenes predichas como {pais}"):
    st.session_state["filtro_galeria"] = {"real": [], "pred": [pais], "acierto": "Todos"}
    st.switch_page("pages/3_Galeria.py")
