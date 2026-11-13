import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

import comun
import mapas
from explorer.core.datos import leer_imagen
from explorer.core.gradcam import gradcam, superponer
from explorer.core.modelo import normalizacion

POR_PAGINA = 24
COLUMNAS = 6

ctx = comun.ctx()
df, config, modelo, clases = ctx["df"], ctx["config"], ctx["modelo"], ctx["clases"]
es_cluster = ctx["tarea"] == "cluster"
nombre = ctx["nombre_clase"]
modo = ctx["tarea"]  # las claves de los filtros dependen de la tarea (opciones distintas)
continentes = sorted(df["continente_real"].dropna().unique())

# Filtro precargado desde otra página (se aplica una sola vez).
inicial = st.session_state.pop("filtro_galeria", None)
if inicial:
    st.session_state[f"g_real_{modo}"] = inicial["real"]
    st.session_state[f"g_pred_{modo}"] = inicial["pred"]
    st.session_state["g_acierto"] = inicial["acierto"]
    st.session_state["g_pagina"] = 1
    st.session_state.pop("g_seleccion", None)


@st.cache_data(show_spinner=False, max_entries=512)
def miniatura(ruta, idx):
    im = leer_imagen(ruta, idx)
    im.thumbnail((320, 320))
    return im


st.title("Galería y Grad-CAM")
with st.expander("Filtros", expanded=True):
    f = st.columns(5 if es_cluster else 4)
    real = f[0].multiselect(f"{nombre.capitalize()} real", clases, key=f"g_real_{modo}")
    pred = f[1].multiselect(f"{nombre.capitalize()} predicho", clases, key=f"g_pred_{modo}")
    cont_real = f[2].multiselect("Continente real", continentes, key="g_cont_real")
    acierto = f[3].radio(f"Acierto de {nombre}", ["Todos", "Aciertos", "Errores"], horizontal=True,
                         key="g_acierto")
    paises_sel = (f[4].multiselect("País real", sorted(df["pais_real"].dropna().unique()), key="g_pais")
                  if es_cluster else [])
    g = st.columns(4 if es_cluster else 3)
    conf = g[0].slider("Confianza", 0.0, 1.0, (0.0, 1.0), 0.01, key="g_conf")
    ordenes = ["Errores más seguros primero", "Más confianza", "Menos confianza", "Aleatorio"]
    if es_cluster:
        ordenes.insert(1, "Mayor error (km)")
    orden = g[1].selectbox("Orden", ordenes, key=f"g_orden_{modo}")
    mismo_cont = g[2].selectbox("Error entre continentes", ["Cualquiera", "Mismo continente",
                                                           "Distinto continente"], key="g_mismo")
    if es_cluster:
        tope = float(np.ceil(df["error_km"].max() / 500) * 500)
        rango_km = g[3].slider("Error (km)", 0.0, tope, (0.0, tope), 50.0, key="g_km")

d = df[df["en_modelo"]]
if real:
    d = d[d["clase_real"].isin(real)]
if pred:
    d = d[d["clase_predicha"].isin(pred)]
if cont_real:
    d = d[d["continente_real"].isin(cont_real)]
if paises_sel:
    d = d[d["pais_real"].isin(paises_sel)]
if acierto == "Aciertos":
    d = d[d["acierto"]]
elif acierto == "Errores":
    d = d[~d["acierto"]]
if mismo_cont == "Mismo continente":
    d = d[~d["acierto"] & d["acierto_continente"]]
elif mismo_cont == "Distinto continente":
    d = d[~d["acierto_continente"]]
d = d[d["confianza"].between(*conf)]
if es_cluster:
    d = d[d["error_km"].between(*rango_km)]
if orden == "Errores más seguros primero":
    d = d.sort_values(["acierto", "confianza"], ascending=[True, False])
elif orden == "Mayor error (km)":
    d = d.sort_values("error_km", ascending=False)
elif orden == "Más confianza":
    d = d.sort_values("confianza", ascending=False)
elif orden == "Menos confianza":
    d = d.sort_values("confianza")
else:
    d = d.sample(frac=1, random_state=0)

n = len(d)
resumen = f"**{n:,}** imágenes"
if n:
    resumen += f" · accuracy de {nombre} del filtro: {d['acierto'].mean():.1%}"
    if es_cluster:
        resumen += f" · error mediano {d['error_km'].median():,.0f} km"
st.write(resumen)
if n == 0:
    st.stop()


def pie(fila):
    texto = f"{fila['clase_real']} → {fila['clase_predicha']} ({fila['confianza']:.0%})"
    if es_cluster:
        texto += f" · {fila['pais_real']} · {fila['error_km']:,.0f} km"
    return texto


paginas = max(1, -(-n // POR_PAGINA))
pagina = st.number_input("Página", 1, paginas, key="g_pagina")
parte = d.iloc[(pagina - 1) * POR_PAGINA: pagina * POR_PAGINA]
columnas = st.columns(COLUMNAS)
for k, (_, fila) in enumerate(parte.iterrows()):
    with columnas[k % COLUMNAS]:
        st.image(miniatura(ctx["ruta_parquet"], int(fila["idx_fila"])), caption=pie(fila))
        if st.button("Ver", key=f"ver_{fila['idx_fila']}", width="stretch"):
            st.session_state["g_seleccion"] = int(fila["idx_fila"])

# ---- Detalle + Grad-CAM ----
idx = st.session_state.get("g_seleccion")
if idx is None or idx >= len(df):
    st.stop()
fila = df.loc[df["idx_fila"] == idx].iloc[0]
st.divider()
st.subheader(f"Imagen {fila['image_id']} · real {fila['clase_real']} · predicho {fila['clase_predicha']}")

imagen = leer_imagen(ctx["ruta_parquet"], idx)
tensor = ctx["transform"](imagen)
media, std = (np.asarray(v, dtype=np.float32)[:, None, None] for v in normalizacion(config))
recorte = np.clip(tensor.numpy() * std + media, 0, 1).transpose(1, 2, 0)  # lo que ve el modelo
recorte_pil = Image.fromarray((recorte * 255).astype(np.uint8))

probs = ctx["res"]["probs"][idx].astype(np.float32)
top = np.argsort(-probs)[:5]
c1, c2 = st.columns([2, 1])
with c1:
    st.image(imagen, caption="Original", width="stretch")
with c2:
    fig = px.bar(pd.DataFrame({"clase": [clases[i] for i in top], "prob": probs[top]}),
                 x="prob", y="clase", orientation="h", title="Top-5")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    texto = f"Fecha: {fila.get('date')} · lat/lon: {fila['latitude']:.4f}, {fila['longitude']:.4f}"
    if es_cluster:
        texto += (f" · país {fila['pais_real']} · error {fila['error_km']:,.0f} km "
                  f"(cota {fila['error_km_cota']:,.0f} km)")
    st.caption(texto)
    if es_cluster:
        # Ubicación real (estrella) y los 5 prototipos más probables (tamaño según probabilidad).
        clusters = ctx["clusters"]
        mapa = mapas.figura(alto=300, centro=(fila["latitude"], fila["longitude"]), escala=1.6,
                            revision=int(idx))
        mapas.bordes(mapa, ctx["aristas"], color="#888", ancho=0.6)
        mapas.contorno(mapa, ctx["contornos"][int(fila["cluster_real"])], ancho=2)
        tops = clusters.iloc[top]
        mapa.add_trace(go.Scattergeo(
            lat=tops["lat"], lon=tops["lon"], mode="markers+text", text=[e.split(" ")[0] for e in tops["etiqueta"]],
            textposition="top center", hoverinfo="text", showlegend=False,
            hovertext=[f"{e}: {p:.1%}" for e, p in zip(tops["etiqueta"], probs[top])],
            marker=dict(size=6 + 30 * probs[top] / probs[top].max(), color="#1f77b4",
                        line=dict(color="white", width=1))))
        mapa.add_trace(go.Scattergeo(lat=[fila["latitude"]], lon=[fila["longitude"]], mode="markers",
                                     marker=dict(symbol="star", size=16, color="#d62728",
                                                 line=dict(color="black", width=1)),
                                     hovertext=["ubicación real"], hoverinfo="text", showlegend=False))
        st.plotly_chart(mapa, width="stretch", config={"scrollZoom": True})
        st.caption("Estrella: ubicación real · contorno rojo: su cluster · azul: top-5 predichos.")
    else:
        st.map(pd.DataFrame({"lat": [fila["latitude"]], "lon": [fila["longitude"]]}), height=180)

st.markdown("#### Grad-CAM")
clases_default = []
if fila["clase_predicha"] in clases:
    clases_default.append(fila["clase_predicha"])
if fila["clase_real"] in clases and fila["clase_real"] not in clases_default:
    clases_default.append(fila["clase_real"])
objetivos = st.multiselect("Clases a explicar", clases, default=clases_default, key=f"cam_{modo}_{idx}")
alpha = st.slider("Opacidad del mapa", 0.1, 0.9, 0.5, 0.05)
if objetivos:
    cols = st.columns(len(objetivos) + 1)
    cols[0].image(recorte_pil, caption="Entrada (recorte 224)", width="stretch")
    for col, p in zip(cols[1:], objetivos):
        c = clases.index(p)
        mapa_cam = gradcam(modelo, config, tensor, c, ctx["device"])
        col.image(superponer(recorte_pil, mapa_cam, alpha),
                  caption=f"{p} (prob {probs[c]:.1%})", width="stretch")
