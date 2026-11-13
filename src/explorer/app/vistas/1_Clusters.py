import pandas as pd
import plotly.express as px
import streamlit as st

import comun
import mapas
from explorer.core.metricas import resumen_por_cluster

NINGUNO = "(ninguno: ver todos)"

ctx = comun.ctx()
df, clusters = ctx["df"], ctx["clusters"]
resumen = resumen_por_cluster(df, clusters)
etiquetas = list(resumen.index)
d = df[df["en_modelo"]]

st.title("Clusters")
st.caption(f"{len(clusters)} celdas de Voronoi sobre la esfera, una por prototipo. Nombre = C<id> + "
           f"país dominante en `{ctx['fuente_clusters']}`. Hacé click en un prototipo o un punto del mapa, "
           "o en una fila de la tabla, para seleccionar su cluster.")


# ---- selección: selectbox, click en el mapa o fila de la tabla (gana el último cambio) ----
def _desde_evento(clave_evento, a_etiqueta):
    evento = st.session_state.get(clave_evento)
    seleccion = getattr(evento, "selection", None) or (evento or {}).get("selection") or {}
    firma = repr(seleccion)
    if st.session_state.get(f"_firma_{clave_evento}") == firma:
        return
    st.session_state[f"_firma_{clave_evento}"] = firma
    elegido = a_etiqueta(seleccion)
    if elegido in etiquetas:
        st.session_state["cl_sel"] = elegido


def _de_mapa(sel):
    puntos = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", [])
    for p in puntos:
        dato = p.get("customdata")
        dato = dato[0] if isinstance(dato, (list, tuple)) else dato
        if dato in etiquetas:
            return dato
    return None


def _de_tabla(sel):
    filas = sel.get("rows", []) if isinstance(sel, dict) else getattr(sel, "rows", [])
    return etiquetas[filas[0]] if filas else None


_desde_evento("cl_mapa", _de_mapa)
_desde_evento("cl_tabla", _de_tabla)
if st.session_state.get("cl_sel") not in [NINGUNO, *etiquetas]:
    st.session_state["cl_sel"] = NINGUNO

c = st.columns([2, 1.3, 1, 1, 1])
sel = c[0].selectbox("Cluster", [NINGUNO, *etiquetas], key="cl_sel")
color_por = c[1].selectbox("Color de los prototipos", ["recall", "error_mediano_km", "precision", "n",
                                                       "continente", "cluster"], key="cl_color")
imagenes = c[2].selectbox("Imágenes en el mapa", ["Ninguna", "Del cluster", "Todas"], index=1,
                          key="cl_imgs", help="Puntos del dataset evaluado, ubicados en su lugar real.")
etiquetas_mapa = c[3].checkbox("Rótulos", value=True, key="cl_rotulos")
acercar = c[4].checkbox("Acercar", value=True, key="cl_acercar", help="Centrar el mapa en el cluster elegido.")
elegido = None if sel == NINGUNO else sel

# ---- mapa ----
if elegido:
    fila = resumen.loc[elegido]
    k = int(fila["ind_cluster"])
    flujo = st.radio("Flechas", ["Real → predicho (a dónde se va el error)",
                                 "Predicho ← real (de dónde vienen los falsos positivos)"],
                     horizontal=True, key="cl_flujo")
    hacia = flujo.startswith("Real")
centro = (fila["lat"], fila["lon"]) if elegido and acercar else None
fig = mapas.figura(alto=620, centro=centro, escala=2.2 if centro else 1.0, revision=f"{elegido}-{acercar}")

if imagenes == "Todas":
    colores = [mapas.color_cluster(x) for x in d["cluster_real"]]
    mapas.puntos_imagenes(fig, d, colores, "imágenes", tamano=3, opacidad=0.45)
mapas.bordes(fig, ctx["aristas"])
if elegido:
    mapas.contorno(fig, ctx["contornos"][k])
    if imagenes != "Ninguna":
        propias = d[d["cluster_real"] == k]
        for ok, nombre, color in [(True, "acierto", "#2ca02c"), (False, "error", "#d62728")]:
            sub = propias[propias["acierto"] == ok]
            if len(sub):
                mapas.puntos_imagenes(fig, sub, color, f"imágenes del cluster: {nombre}", tamano=6,
                                      opacidad=0.85, mostrar_leyenda=True)
    if hacia:
        conteo = d[(d["cluster_real"] == k) & ~d["acierto"]]["clase_predicha"].value_counts()
        total, color, nombre = int(fila["n"] or 0), "#1f77b4", "predicho cuando era este"
    else:
        conteo = d[(d["cluster_predicho"] == k) & ~d["acierto"]]["clase_real"].value_counts()
        total, color, nombre = int(fila["n_predichas"] or 0), "#ff7f0e", "real cuando predijo este"
    destinos = resumen.loc[conteo.index[:10], ["lat", "lon"]].assign(n=conteo.values[:10])
    destinos["texto"] = [f"{e}: {n} imgs ({n / max(total, 1):.0%})" for e, n in zip(destinos.index, destinos["n"])]
    mapas.flujos(fig, (fila["lat"], fila["lon"]), destinos, color, nombre)
mapas.prototipos(fig, resumen, color_por, etiquetas=etiquetas_mapa, resaltado=elegido)
st.plotly_chart(fig, width="stretch", key="cl_mapa", on_select="rerun", selection_mode="points",
                config={"scrollZoom": True})

# ---- tabla ----
columnas = ["ind_cluster", "continente", "paises", "n", "n_predichas", "recall", "precision", "f1", "top5",
            "error_mediano_km", "cota_mediana_km", "conf_aciertos", "conf_errores", "se_confunde_con",
            "falsos_positivos_de"]
st.dataframe(
    resumen[columnas], width="stretch", height=380, key="cl_tabla", on_select="rerun",
    selection_mode="single-row",
    column_config={
        "recall": st.column_config.ProgressColumn("recall", min_value=0, max_value=1, format="%.2f"),
        "precision": st.column_config.NumberColumn(format="%.2f"),
        "f1": st.column_config.NumberColumn(format="%.2f"),
        "top5": st.column_config.NumberColumn(format="%.2f"),
        "error_mediano_km": st.column_config.NumberColumn("mediana km", format="%.0f"),
        "cota_mediana_km": st.column_config.NumberColumn("cota km", format="%.0f",
                                                         help="Distancia mediana foto → su propio prototipo."),
        "conf_aciertos": st.column_config.NumberColumn(format="%.2f"),
        "conf_errores": st.column_config.NumberColumn(format="%.2f"),
    })

if not elegido:
    st.stop()

# ---- detalle del cluster ----
st.divider()
st.subheader(f"{elegido} · {fila['continente']}")
st.caption(f"Prototipo en {fila['lat']:.2f}, {fila['lon']:.2f} · composición del train: {fila['paises']}")
m = st.columns(5)
m[0].metric("Imágenes reales", f"{int(fila['n'] or 0):,}")
m[1].metric("Recall", f"{fila['recall']:.1%}" if pd.notna(fila["recall"]) else "–")
m[2].metric("Precisión", f"{fila['precision']:.1%}" if pd.notna(fila["precision"]) else "–")
m[3].metric("Error mediano", f"{fila['error_mediano_km']:,.0f} km" if pd.notna(fila["error_mediano_km"]) else "–")
m[4].metric("Cota mediana", f"{fila['cota_mediana_km']:,.0f} km" if pd.notna(fila["cota_mediana_km"]) else "–",
            help="Error si siempre se acertara la celda: cuán grande es el cluster.")


def barras(serie, titulo, color):
    serie = serie[serie > 0].sort_values(ascending=False).head(15)
    if serie.empty:
        st.write("Sin casos.")
        return
    fig = px.bar(pd.DataFrame({"clase": serie.index, "n": serie.values}), x="clase", y="n", title=titulo)
    fig.update_traces(marker_color=color)
    fig.update_layout(xaxis_title=None, yaxis_title="imágenes", height=340)
    st.plotly_chart(fig, width="stretch")


reales, predichas = d[d["cluster_real"] == k], d[d["cluster_predicho"] == k]
izq, der = st.columns(2)
with izq:
    st.markdown(f"**Cuando es {elegido}** ({len(reales)} imgs), el modelo predice:")
    barras(reales["clase_predicha"].value_counts(), f"Real = {elegido} → predicho", "#4C78A8")
with der:
    st.markdown(f"**Cuando predice {elegido}** ({len(predichas)} imgs), en realidad era:")
    barras(predichas["clase_real"].value_counts(), f"Predicho = {elegido} ← real", "#E45756")

if len(reales):
    st.markdown("**Países reales dentro del cluster** (dataset evaluado)")
    por_pais = reales.groupby("pais_real").agg(
        n=("acierto", "size"), acc_cluster=("acierto", "mean"), top5=("acierto_top5", "mean"),
        error_mediano_km=("error_km", "median")).sort_values("n", ascending=False)
    st.dataframe(por_pais, width="stretch", column_config={
        "acc_cluster": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "top5": st.column_config.NumberColumn(format="%.2f"),
        "error_mediano_km": st.column_config.NumberColumn(format="%.0f")})

b1, b2, _ = st.columns([1, 1, 2])
if b1.button(f"Ver imágenes reales de {elegido}"):
    st.session_state["filtro_galeria"] = {"real": [elegido], "pred": [], "acierto": "Todos"}
    st.switch_page("vistas/3_Galeria.py")
if b2.button(f"Ver imágenes predichas como {elegido}"):
    st.session_state["filtro_galeria"] = {"real": [], "pred": [elegido], "acierto": "Todos"}
    st.switch_page("vistas/3_Galeria.py")
