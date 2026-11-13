import pandas as pd
import streamlit as st

import comun
from explorer.core.clusters import UMBRALES_KM
from explorer.core.metricas import resumen_km, resumen_por_continente, resumen_por_grupo

ctx = comun.ctx()
df, config = ctx["df"], ctx["config"]
d = df[df["en_modelo"]]

st.title("Explorador de rendimiento")
st.caption(f"{ctx['corrida']['etiqueta']}  |  {ctx['dataset']['etiqueta']}")
formato = {c: "{:.1%}" for c in ["acc", "top5", "acc_continente", "acc_pais"]}
formato |= {c: "{:,.0f}" for c in ["error_mediano_km", "cota_mediana_km"]}

if ctx["tarea"] == "pais":
    c = st.columns(4)
    c[0].metric("Imágenes evaluadas", f"{len(d):,}")
    c[1].metric("Accuracy país", f"{d['acierto'].mean():.2%}")
    c[2].metric("Accuracy continente", f"{d['acierto_continente'].mean():.2%}")
    c[3].metric("Top-5 país", f"{d['acierto_top5'].mean():.2%}")
    st.caption(f"Países en el modelo: {len(config['paises'])} · imagen {config['tamano_imagen']}px · "
               f"entrenado con {config.get('filas_train', '?')} imágenes")
else:
    km, km_mbr = resumen_km(d["error_km"]), resumen_km(d["error_km_mbr"])
    c = st.columns(5)
    c[0].metric("Imágenes evaluadas", f"{len(d):,}")
    c[1].metric("Accuracy cluster", f"{d['acierto'].mean():.2%}")
    c[2].metric("Top-5 cluster", f"{d['acierto_top5'].mean():.2%}")
    c[3].metric("Error mediano", f"{km['mediana_km']:,.0f} km")
    c[4].metric("Error medio", f"{km['media_km']:,.0f} km")
    st.markdown("**Imágenes dentro de cada distancia** (prototipo predicho → ubicación real)")
    u = st.columns(len(UMBRALES_KM))
    for col, umbral in zip(u, UMBRALES_KM):
        col.metric(f"≤ {umbral:,} km", f"{km[f'<= {umbral:,} km']:.1%}")
    st.caption(
        f"{len(ctx['clusters'])} clusters · imagen {config['tamano_imagen']}px · entrenado con "
        f"{config.get('filas_train', '?')} imágenes · {config.get('loss', '')} · "
        f"cota (acertando siempre la celda): mediana {d['error_km_cota'].median():,.0f} km · "
        f"predicción MBR: acc {d['acierto_mbr'].mean():.1%}, mediana {km_mbr['mediana_km']:,.0f} km · "
        f"país dominante del cluster predicho = país real: {d['acierto_pais'].mean():.1%}")

st.subheader("Por continente (real)")
st.dataframe(resumen_por_continente(df).style.format(formato), width="stretch")

if ctx["tarea"] == "cluster":
    with st.expander("Por país (real)"):
        por_pais = resumen_por_grupo(df, "pais_real").sort_values("n", ascending=False)
        st.dataframe(por_pais.style.format(formato), width="stretch", height=420)

historial = ctx["corrida"]["carpeta"] / "historial.csv"
if historial.is_file():
    h = pd.read_csv(historial).set_index("epoca")
    curvas = st.columns(2 if "error_mediano_km_validation" in h else 1)
    with curvas[0]:
        st.subheader("Curvas de loss")
        st.line_chart(h[["loss_train", "loss_validation"]])
    if len(curvas) > 1:
        with curvas[1]:
            st.subheader("Error mediano (km)")
            st.line_chart(h[[c for c in ["error_mediano_km_train", "error_mediano_km_validation"] if c in h]])

pagina = "Clusters" if ctx["tarea"] == "cluster" else "Paises"
st.info(f"Usá las páginas del menú lateral: **{pagina}**, **Confusiones** y **Galeria**.")
