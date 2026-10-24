import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

import comun
from explorer.core.datos import leer_imagen
from explorer.core.gradcam import gradcam, superponer

POR_PAGINA = 24
COLUMNAS = 6

st.set_page_config(page_title="Galería", layout="wide")
ctx = comun.contexto()
df, config, modelo = ctx["df"], ctx["config"], ctx["modelo"]
paises = config["paises"]
continentes = sorted(set(config["pais_a_continente"].values()))

# Filtro precargado desde otra página (se aplica una sola vez).
inicial = st.session_state.pop("filtro_galeria", None)
if inicial:
    st.session_state["g_real"] = inicial["real"]
    st.session_state["g_pred"] = inicial["pred"]
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
    f = st.columns(4)
    real = f[0].multiselect("País real", paises, key="g_real")
    pred = f[1].multiselect("País predicho", paises, key="g_pred")
    cont_real = f[2].multiselect("Continente real", continentes, key="g_cont_real")
    acierto = f[3].radio("Acierto de país", ["Todos", "Aciertos", "Errores"], horizontal=True,
                         key="g_acierto")
    g = st.columns(3)
    conf = g[0].slider("Confianza", 0.0, 1.0, (0.0, 1.0), 0.01, key="g_conf")
    orden = g[1].selectbox("Orden", ["Errores más seguros primero", "Más confianza", "Menos confianza",
                                     "Aleatorio"], key="g_orden")
    mismo_cont = g[2].selectbox("Error entre continentes", ["Cualquiera", "Mismo continente",
                                                           "Distinto continente"], key="g_mismo")

d = df[df["en_modelo"]]
if real:
    d = d[d["pais_real"].isin(real)]
if pred:
    d = d[d["pais_predicho"].isin(pred)]
if cont_real:
    d = d[d["continente_real"].isin(cont_real)]
if acierto == "Aciertos":
    d = d[d["acierto_pais"]]
elif acierto == "Errores":
    d = d[~d["acierto_pais"]]
if mismo_cont == "Mismo continente":
    d = d[~d["acierto_pais"] & d["acierto_continente"]]
elif mismo_cont == "Distinto continente":
    d = d[~d["acierto_continente"]]
d = d[d["confianza"].between(*conf)]
if orden == "Errores más seguros primero":
    d = d.sort_values(["acierto_pais", "confianza"], ascending=[True, False])
elif orden == "Más confianza":
    d = d.sort_values("confianza", ascending=False)
elif orden == "Menos confianza":
    d = d.sort_values("confianza")
else:
    d = d.sample(frac=1, random_state=0)

n = len(d)
st.write(f"**{n:,}** imágenes" + (f" · accuracy de país del filtro: {d['acierto_pais'].mean():.1%}"
                                    if n else ""))
if n == 0:
    st.stop()

paginas = max(1, -(-n // POR_PAGINA))
pagina = st.number_input("Página", 1, paginas, 1, key="g_pagina")
parte = d.iloc[(pagina - 1) * POR_PAGINA: pagina * POR_PAGINA]
columnas = st.columns(COLUMNAS)
for k, (_, fila) in enumerate(parte.iterrows()):
    with columnas[k % COLUMNAS]:
        st.image(miniatura(ctx["ruta_parquet"], int(fila["idx_fila"])),
                 caption=f"{fila['pais_real']} → {fila['pais_predicho']} ({fila['confianza']:.0%})")
        if st.button("Ver", key=f"ver_{fila['idx_fila']}", width="stretch"):
            st.session_state["g_seleccion"] = int(fila["idx_fila"])

# ---- Detalle + Grad-CAM ----
idx = st.session_state.get("g_seleccion")
if idx is None:
    st.stop()
fila = df.loc[df["idx_fila"] == idx].iloc[0]
st.divider()
st.subheader(f"Imagen {fila['image_id']} · real {fila['pais_real']} · predicho {fila['pais_predicho']}")

imagen = leer_imagen(ctx["ruta_parquet"], idx)
tensor = ctx["transform"](imagen)
recorte = (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()  # lo que ve el modelo
recorte_pil = Image.fromarray((recorte * 255).astype(np.uint8))

probs = ctx["res"]["probs"][idx].astype(np.float32)
top = np.argsort(-probs)[:5]
c1, c2 = st.columns([2, 1])
with c1:
    st.image(imagen, caption="Original", width="stretch")
with c2:
    fig = px.bar(pd.DataFrame({"pais": [paises[i] for i in top], "prob": probs[top]}),
                 x="prob", y="pais", orientation="h", title="Top-5")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Fecha: {fila.get('date')} · lat/lon: {fila['latitude']:.4f}, {fila['longitude']:.4f}")
    st.map(pd.DataFrame({"lat": [fila["latitude"]], "lon": [fila["longitude"]]}), height=180)

st.markdown("#### Grad-CAM")
clases_default = []
if fila["pais_predicho"] in paises:
    clases_default.append(fila["pais_predicho"])
if fila["pais_real"] in paises and fila["pais_real"] not in clases_default:
    clases_default.append(fila["pais_real"])
objetivos = st.multiselect("Clases a explicar", paises, default=clases_default, key=f"cam_{idx}")
alpha = st.slider("Opacidad del mapa", 0.1, 0.9, 0.5, 0.05)
if objetivos:
    cols = st.columns(len(objetivos) + 1)
    cols[0].image(recorte_pil, caption="Entrada (recorte 224)", width="stretch")
    for col, p in zip(cols[1:], objetivos):
        c = paises.index(p)
        mapa = gradcam(modelo, config, tensor, c, ctx["device"])
        col.image(superponer(recorte_pil, mapa, alpha),
                  caption=f"{p} (prob {probs[c]:.1%})", width="stretch")
