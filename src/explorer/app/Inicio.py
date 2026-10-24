import pandas as pd
import streamlit as st

import comun
from explorer.core.metricas import resumen_por_continente

st.set_page_config(page_title="Explorador de modelo", layout="wide")
ctx = comun.contexto()
df, config = ctx["df"], ctx["config"]
d = df[df["en_modelo"]]

st.title("Explorador de rendimiento")
st.caption(f"{ctx['corrida']['etiqueta']}  |  {ctx['dataset']['etiqueta']}")

c = st.columns(4)
c[0].metric("Imágenes evaluadas", f"{len(d):,}")
c[1].metric("Accuracy país", f"{d['acierto_pais'].mean():.2%}")
c[2].metric("Accuracy continente", f"{d['acierto_continente'].mean():.2%}")
c[3].metric("Top-5 país", f"{d['acierto_top5'].mean():.2%}")
st.caption(f"Países en el modelo: {len(config['paises'])} · imagen {config['tamano_imagen']}px · "
           f"entrenado con {config.get('filas_train', '?')} imágenes")

st.subheader("Por continente (real)")
por_cont = resumen_por_continente(df)
st.dataframe(por_cont.style.format({"acc_pais": "{:.1%}", "acc_continente": "{:.1%}",
                                    "top5": "{:.1%}"}), width="stretch")

historial = ctx["corrida"]["carpeta"] / "historial.csv"
if historial.is_file():
    st.subheader("Curvas de loss")
    h = pd.read_csv(historial).set_index("epoca")
    st.line_chart(h[["loss_train", "loss_validation"]])

st.info("Usá las páginas del menú lateral: **Paises**, **Confusiones** y **Galeria**.")
