"""Figuras y datos de la presentación final. Todo sale de archivos del repo, salvo los dos ensambles de snapshots de
20k, que se copian de sus notebooks (no guardan un json de métricas).

Uso (desde la raíz del repo): .venv/Scripts/python.exe presentacion/armar_figuras.py
Escribe presentacion/figuras/*.png|jpg y presentacion/figuras/datos_graficos.json (datos de los gráficos nativos).
Necesita: logits de test (scripts/inferir_test.py), la copia local del dataset (../world-streetview-500k/data) y las
salidas guardadas de los notebooks del EDA.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm  # noqa: E402
from PIL import Image  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.entrenamiento.datos import leer_filas  # noqa: E402
from src.entrenamiento.entrenamiento import haversine_np, softmax_np, to_xyz  # noqa: E402

SALIDA = RAIZ / "presentacion" / "figuras"
SALIDA.mkdir(parents=True, exist_ok=True)
DATOS = RAIZ.parent / "world-streetview-500k" / "data"
E2M = RAIZ / "outputs" / "entrenamiento_2m"

# Paleta de la presentación (mismos tonos que el tema del pptx) y colores fijos por modelo.
TINTA, SECUNDARIO, TENUE = "#16202b", "#4f5b69", "#8a94a0"
TIERRA, AGUA, BORDE = "#e9e6df", "#ffffff", "#ffffff"
PIN_REAL, PIN_MODELO = "#10263b", "#e4572e"
MODELOS = {"baseline_celdas_2m": ("Celdas CE", "#2a78d6"), "baseline_celdas_aug_2m": ("Celdas CE + aug", "#eb6834"),
           "completo_2m": ("Celdas completo", "#1baf7a"), "baseline_paises101_2m": ("Países 101", "#eda100")}
SECUENCIAL = LinearSegmentedColormap.from_list("seq", ["#eef3f8", "#9ec3ef", "#2a78d6", "#0d3a73"])

# Tamaños pensados para proyectar en un aula.
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "font.size": 16, "axes.labelsize": 16,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": TENUE, "axes.labelcolor": SECUNDARIO,
                     "xtick.color": SECUNDARIO, "ytick.color": SECUNDARIO, "axes.titlecolor": TINTA,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.7, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})
MUNDO = gpd.read_file(RAIZ / "src/notebooks/eda/assets/ne_110m_admin_0_countries.geojson")
MUNDO = MUNDO[MUNDO.CONTINENT != "Antarctica"]
datos_graficos: dict = {}


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre)
    plt.close(fig)
    print("  ", nombre)


def mapa_base(eje, extension=(-170, 180, -58, 80)):
    MUNDO.plot(ax=eje, color=TIERRA, edgecolor=BORDE, linewidth=0.4)
    eje.set_xlim(extension[0], extension[1])
    eje.set_ylim(extension[2], extension[3])
    eje.set_axis_off()


# ------------------------------------------------------------------ datos de test
test = pd.read_parquet(E2M / "test_filas.parquet")
_, pano = np.unique(test.panoid.to_numpy(), return_inverse=True)
n_pano = pano.max() + 1
primera = np.full(n_pano, -1)
primera[pano[::-1]] = np.arange(len(test))[::-1]
proto = pd.read_csv(RAIZ / "src/notebooks/preprocessing/assets/prototipos_train-completo_k70.csv").sort_values(
    "ind_cluster")[["lat", "lon"]].to_numpy()


def leer_imagenes(indices):
    """Imágenes de test por índice de fila de test_filas (PIL), en el mismo orden."""
    sel = test.loc[list(indices)]
    salida = {}
    for archivo, g in sel.groupby("archivo"):
        for datos, i in leer_filas(DATOS / archivo, g.fila.to_numpy(), g.index.to_numpy()):
            salida[i] = Image.open(io.BytesIO(datos)).convert("RGB")
    return [salida[i] for i in indices]


def indices_pano(panoid):
    return list(test.index[test.panoid == panoid])


# ------------------------------------------------------------------ fotos
print("Fotos")
PORTADA = ["jL6CMhdCSUmKrV-QT2Caug", "4ngudG9Qovjr6R3DhU3k3w", "ySrmyeY6NriVlSPvwUawyQ", "wIJqdIpSa1HQTvPCnK6ruw",
           "Ah5ekEIo-4LTWyJcEXfcPA", "xF8vNxgXvsj0WnhrG84ZkQ"]
for k, im in enumerate(leer_imagenes([indices_pano(p)[0] for p in PORTADA]), 1):
    im.save(SALIDA / f"portada_{k}.jpg", quality=90)
VISTAS = "jL6CMhdCSUmKrV-QT2Caug"  # Noruega: 4 vistas de un panorama.
for k, im in enumerate(leer_imagenes(indices_pano(VISTAS)), 1):
    im.save(SALIDA / f"vista_{k}.jpg", quality=90)

# Ronda de GeoGuessr con el modelo ganador (Celdas CE, 4 vistas promediadas).
RONDA = "4ngudG9Qovjr6R3DhU3k3w"  # Brasil.
lg_ce = np.load(E2M / "baseline_celdas_2m/logits/mejor__test.npy")
filas = indices_pano(RONDA)
leer_imagenes(filas[:1])[0].save(SALIDA / "ronda_foto.jpg", quality=90)
celda = softmax_np(lg_ce[filas].mean(0, keepdims=True))[0].argmax()
lat_r, lon_r = test.latitude[filas[0]], test.longitude[filas[0]]
km_r = float(haversine_np(lat_r, lon_r, proto[celda, 0], proto[celda, 1]))
fig, eje = plt.subplots(figsize=(6.4, 6.4))
mapa_base(eje, (-82, -32, -36, 8))
eje.plot([lon_r, proto[celda, 1]], [lat_r, proto[celda, 0]], color=TINTA, lw=1.6, ls=(0, (4, 3)))
eje.scatter([lon_r], [lat_r], s=170, color=PIN_REAL, zorder=5, edgecolor="white", linewidth=2)
eje.scatter([proto[celda, 1]], [proto[celda, 0]], s=170, color=PIN_MODELO, zorder=5, edgecolor="white", linewidth=2)
eje.annotate("ubicación real", (lon_r, lat_r), xytext=(10, -16), textcoords="offset points", color=PIN_REAL,
             fontsize=18, fontweight="bold")
eje.annotate("predicción del modelo", (proto[celda, 1], proto[celda, 0]), xytext=(-12, 12),
             textcoords="offset points", color=PIN_MODELO, fontsize=18, fontweight="bold", ha="right")
guardar(fig, "ronda_mapa.png")
datos_graficos["ronda"] = {"km": round(km_r), "puntaje": round(5000 * np.exp(-km_r / 1492.7))}

# ------------------------------------------------------------------ EDA
print("EDA")
splits = pq.read_table(RAIZ / "datasets/manifest_2m/splits.parquet",
                       columns=["split", "panoid", "country_code"]).to_pandas()
todo = splits[splits.split.isin(["train", "val", "test"])]
paises = todo.country_code.astype(str).str.strip().str.upper()
conteo = paises.value_counts()
datos_graficos["dataset"] = {"imagenes": int(len(todo)), "panoramas": int(todo.panoid.nunique()),
                             "paises": int(conteo.size)}
top = (conteo / conteo.sum() * 100).head(12)
datos_graficos["paises_top"] = {"etiquetas": list(top.index), "valores": [round(v, 1) for v in top.values],
                                "resto": round(100 - top.sum(), 1), "top3": round(top.head(3).sum(), 1),
                                "paises_menos_1000": int((conteo < 1000).sum())}

coords = pq.read_table(RAIZ / "datasets/manifest_2m/manifest.parquet", columns=["latitude", "longitude"]).to_pandas()
fig, eje = plt.subplots(figsize=(11, 5.2))
mapa_base(eje)
hb = eje.hexbin(coords.longitude, coords.latitude, gridsize=(180, 60), bins="log", cmap=SECUENCIAL, mincnt=1,
                linewidths=0, extent=(-170, 180, -58, 80))
barra = fig.colorbar(hb, ax=eje, shrink=0.6, pad=0.01)
barra.set_label("imágenes (escala log)")
barra.outline.set_visible(False)
# Código de los países con más imágenes, sobre el mapa (posiciones a mano, dentro de cada país).
from matplotlib import patheffects  # noqa: E402

POS_CODIGO = {"US": (-99, 38), "IN": (78, 21), "BR": (-51, -9), "FR": (1, 47.5), "CA": (-108, 57),
              "AU": (134, -25), "RU": (95, 61), "TH": (101, 15.5), "ID": (113, -2), "JP": (139, 38.5),
              "MX": (-103, 23), "IT": (13, 42.5)}
for codigo in top.index:
    if codigo in POS_CODIGO:
        eje.text(*POS_CODIGO[codigo], codigo, fontsize=16, fontweight="bold", color=TINTA, ha="center", va="center",
                 zorder=6, path_effects=[patheffects.withStroke(linewidth=3.5, foreground="white")])
guardar(fig, "eda_mapa_densidad.png")

nb = json.loads((RAIZ / "src/notebooks/eda/main.ipynb").read_text(encoding="utf-8"))


def imagen_notebook(celda, k=0):
    pngs = [o["data"]["image/png"] for o in nb["cells"][celda].get("outputs", []) if "image/png" in o.get("data", {})]
    return Image.open(io.BytesIO(base64.b64decode(pngs[k])))


pca_px = imagen_notebook(18)
pca_px.crop((0, 0, pca_px.width, int(pca_px.height * 0.633))).save(SALIDA / "pca_pixeles.png")
pca_clip = imagen_notebook(32)
pca_clip.crop((0, int(pca_clip.height * 0.035), int(pca_clip.width * 0.505), int(pca_clip.height * 0.74))).save(
    SALIDA / "pca_clip.png")

# ------------------------------------------------------------------ celdas
print("Celdas")
lon_g, lat_g = np.meshgrid(np.linspace(-180, 180, 1440), np.linspace(-58, 82, 560))
etiqueta = (to_xyz(lat_g.ravel(), lon_g.ravel()) @ to_xyz(proto[:, 0], proto[:, 1]).T).argmax(1).reshape(lat_g.shape)
# Coloreo greedy para que celdas vecinas tengan distinto color.
vecinas = {c: set() for c in range(len(proto))}
for a, b in [(etiqueta[:, :-1], etiqueta[:, 1:]), (etiqueta[:-1], etiqueta[1:])]:
    m = a != b
    for x, y in set(zip(a[m].tolist(), b[m].tolist())):
        vecinas[x].add(y)
        vecinas[y].add(x)
suaves = ["#cfe0f5", "#f7d4c4", "#cdeee0", "#fbe7b5", "#f4cfe0", "#d9d3f5"]
color_celda = {}
for c in sorted(vecinas, key=lambda c: -len(vecinas[c])):
    usados = {color_celda.get(v) for v in vecinas[c]}
    color_celda[c] = next(i for i in range(len(suaves)) if i not in usados)
fig, eje = plt.subplots(figsize=(11, 5.2))
tierra = MUNDO.dissolve()
eje.imshow(np.vectorize(color_celda.get)(etiqueta), extent=(-180, 180, -58, 82), origin="lower",
           cmap=ListedColormap(suaves), vmin=0, vmax=len(suaves) - 1, interpolation="nearest")
# Océano encima para que las celdas solo se vean sobre tierra.
from shapely.geometry import box  # noqa: E402

eje.contour(lon_g, lat_g, etiqueta, levels=np.arange(len(proto) + 1) - 0.5, colors=TINTA, linewidths=0.6)
gpd.GeoSeries([box(-180, -58, 180, 82).difference(tierra.geometry.iloc[0])]).plot(ax=eje, color="white", zorder=3)
MUNDO.boundary.plot(ax=eje, color="#9aa3ad", linewidth=0.35, zorder=4)
eje.scatter(proto[:, 1], proto[:, 0], s=10, color=TINTA, zorder=5)
eje.set_xlim(-170, 180)
eje.set_ylim(-58, 80)
eje.set_axis_off()
guardar(fig, "mapa_celdas.png")

# ------------------------------------------------------------------ recorrido con 20k
print("Recorrido 20k")


def met20k(carpeta):
    ruta = next((RAIZ / "outputs" / carpeta).glob("*/metricas_validation.json"))
    m = json.loads(ruta.read_text(encoding="utf-8"))
    return round(m["accuracy_top1"] * 100, 1), round(m["error_mediano_km"])


aug = sorted((RAIZ / "outputs/resnet18_IMAGENET_celdas_k70_haversine_aug").glob("*/metricas_validation.json"))[0]
m_aug = json.loads(aug.read_text(encoding="utf-8"))  # La primera corrida (semilla 42); la otra es la de semilla 105.
pasos = [("Desde cero", met20k("resnet18_celdas_balanceado")),
         ("ImageNet", met20k("resnet18_IMAGENET_celdas_balanceado")),
         ("Celdas K=70\ndel train completo", met20k("resnet18_IMAGENET_celdas_k70_train-completo")),
         ("Loss\nhaversine", met20k("resnet18_IMAGENET_celdas_k70_haversine")),
         ("Data\naugmentation", (round(m_aug["accuracy_top1"] * 100, 1), round(m_aug["error_mediano_km"]))),
         # Los ensambles de snapshots no guardan un json: valores de sus notebooks (val global, mismas métricas).
         ("Snapshot\nensemble", (34.3, 846)),  # cell_classifier_k70_haversine_aug_snapshot_ensemble.ipynb
         ("SAM", met20k("resnet18_IMAGENET_celdas_k70_haversine_aug_sam")),
         ("SAM +\nsnapshots", (36.2, 805))]  # cell_classifier_k70_haversine_aug_sam_snapshot_ensemble.ipynb
datos_graficos["recorrido_20k"] = {"etiquetas": [p for p, _ in pasos], "top1": [v[0] for _, v in pasos],
                                   "km": [v[1] for _, v in pasos]}

# Overfitting: 20k sin augmentation contra 2M (CE simple).
h20 = pd.read_csv(next((RAIZ / "outputs/resnet18_IMAGENET_celdas_k70_haversine").glob("*/historial.csv")))
h2m = pd.read_csv(E2M / "baseline_celdas_2m/historial.csv")
h2m = h2m[h2m.fase == "base"]
datos_graficos["overfitting"] = {
    "20k": {"x": h20.epoca.tolist(), "train": (h20.accuracy_train * 100).round(1).tolist(),
            "val": (h20.accuracy_validation * 100).round(1).tolist()},
    "2m": {"x": list(range(1, len(h2m) + 1)), "train": (h2m.top1_train * 100).round(1).tolist(),
           "val": (h2m.val_top1 * 100).round(1).tolist()}}
datos_graficos["mas_datos_paises"] = {"etiquetas": ["20k", "50k"], "valores": [
    round(json.loads(next((RAIZ / "outputs/resnet18").glob("20261018*/metricas_validation.json")).read_text())[
        "accuracy_pais"] * 100, 1),
    round(json.loads(next((RAIZ / "outputs/resnet18_50k").glob("*/metricas_validation.json")).read_text())[
        "accuracy_pais"] * 100, 1)]}
# País contra celdas con 20k: misma muestra (20k balanceada por país), misma validación (3.982 imágenes),
# ResNet-18 desde cero. El país predicho se ubica en el medoide del país en el train de 20k.
pred_pais = pd.read_csv(next((RAIZ / "outputs/resnet18").glob("20261018*/predicciones_validation.csv")),
                        keep_default_na=False)  # "NA" es Namibia, no un faltante.
pred_celda = pd.read_csv(next((RAIZ / "outputs/resnet18_voronoi_cells").glob("*/predicciones_validation.csv")),
                         keep_default_na=False)
uni = pred_pais.merge(pred_celda[["image_id", "latitude", "longitude", "error_km"]], on="image_id", validate="one_to_one")
tr20 = pq.read_table(RAIZ / "datasets/samples/muestra_n-20000_umbral-csv-1000_balanceado_paises-71_seed-42_solo-train/"
                     "data/train.parquet", columns=["panoid", "country_code", "latitude", "longitude"]).to_pandas()
medoide = {}
for codigo, g in tr20.drop_duplicates("panoid").groupby("country_code"):
    la, lo = g.latitude.to_numpy(), g.longitude.to_numpy()
    j = haversine_np(la[:, None], lo[:, None], la[None], lo[None]).mean(1).argmin()
    medoide[codigo] = (la[j], lo[j])
pp = np.array([medoide[c] for c in uni.pais_predicho])
km_pais20 = haversine_np(uni.latitude.to_numpy(), uni.longitude.to_numpy(), pp[:, 0], pp[:, 1])
datos_graficos["pais_vs_celdas_20k"] = {
    "imagenes_val": int(len(uni)),
    "acc_pais": round(float(uni.acierto_pais.astype(str).eq("True").mean()) * 100, 1),
    "acc_celdas": round(float(pred_celda.acierto_top1.astype(str).eq("True").mean()) * 100, 1),
    "km_pais": round(float(np.median(km_pais20))), "km_celdas": round(float(uni.error_km.astype(float).median()))}
datos_graficos["escala_datos"] = {"etiquetas": ["20k", "200k", "1.9M"], "km": [
    805, round(json.loads((RAIZ / "outputs/entrenamiento_2m_explorer/prueba_200k_snapshot_largo_3/"
                                  "metricas_validation.json").read_text())["error_mediano_km"]),
    round(json.loads((RAIZ / "outputs/entrenamiento_2m_explorer/completo_2m_snapshot_corto_4/"
                             "metricas_validation.json").read_text())["error_mediano_km"])]}

# ------------------------------------------------------------------ test
print("Test")
pred = {}
for corrida, (nombre, color) in MODELOS.items():
    lg = np.load(next((E2M / corrida / "logits").glob("*__test.npy"))).astype(np.float64)
    if corrida == "baseline_paises101_2m":
        from src.entrenamiento.config import Config  # noqa: E402
        from src.entrenamiento.entrenamiento import clases_y_tablas  # noqa: E402

        cfg = json.loads((E2M / corrida / "config.json").read_text(encoding="utf-8"))
        cfg.pop("huella", None)
        for campo in ("manifest", "splits", "val_global"):
            cfg.pop(campo, None)
        pts, _, _ = clases_y_tablas(Config(**cfg))
    else:
        pts = proto
    suma = np.zeros((n_pano, lg.shape[1]))
    np.add.at(suma, pano, lg)
    for modo, z, filas_m in [("1 vista", lg, np.arange(len(test))), ("4 vistas", suma, primera)]:
        k = z.argmax(1)
        km = haversine_np(test.latitude.to_numpy()[filas_m], test.longitude.to_numpy()[filas_m], pts[k, 0], pts[k, 1])
        pred[(corrida, modo)] = km

# De acá en adelante (resultados finales) se muestran tres modelos: sin "Celdas CE + aug" y con "Países" a secas.
FINALES = {"baseline_celdas_2m": ("Celdas CE simple", "#2a78d6"), "completo_2m": ("Celdas completo", "#1baf7a"),
           "baseline_paises101_2m": ("Países", "#eda100")}


def guardar_fijo(fig, nombre):
    """Sin recorte automático: la imagen conserva el tamaño exacto de la figura (reemplaza una imagen del pptx)."""
    with plt.rc_context({"savefig.bbox": "standard"}):  # bbox_inches=None igual usaría el "tight" global.
        fig.savefig(SALIDA / nombre, dpi=250)
    plt.close(fig)
    print("  ", nombre)


# Distribución acumulada del error. Proporción 2,03:1, la de la imagen en la slide del modelo final.
fig, ejes = plt.subplots(1, 2, figsize=(8.125, 4.0), sharey=True, layout="constrained")
for eje, modo in zip(ejes, ["1 vista", "4 vistas"]):
    for corrida, (nombre, color) in FINALES.items():
        km = np.sort(pred[(corrida, modo)])
        eje.plot(km, np.arange(1, len(km) + 1) / len(km), color=color, lw=2.4, label=nombre)
    for u in [200, 750, 2500]:
        eje.axvline(u, color="#c4c9cf", lw=0.9, ls=":")
        eje.text(u * 1.06, 0.02, f"{u}", color=SECUNDARIO, fontsize=12)
    eje.set_xscale("log")
    eje.set_xlim(30, 20000)
    eje.set_title(modo, loc="left", fontsize=16, fontweight="bold")
    eje.set_xlabel("error en km (escala log)", fontsize=13)
    eje.tick_params(labelsize=12)
ejes[0].set_ylabel("fracción de predicciones", fontsize=13)
manijas, etiquetas = ejes[0].get_legend_handles_labels()
fig.legend(manijas, etiquetas, loc="outside lower center", ncol=3, fontsize=14)
guardar_fijo(fig, "test_cdf.png")

# Barras de la slide de baselines (test, 4 vistas). Proporción 1,39:1, la de las imágenes de esa slide.
for archivo, clave, titulo, formato, tope in [
        ("baselines_puntaje.png", "puntaje", "Puntaje GeoGuessr medio en test (más es mejor)", "{:,.0f}", 4600),
        ("baselines_km.png", "km_mediana", "Error mediano en test (menos es mejor)", "{:,.0f} km", 780)]:
    fig, eje = plt.subplots(figsize=(6.05, 4.35), layout="constrained")
    nombres = [n for n, _ in FINALES.values()][::-1]
    corridas = list(FINALES)[::-1]
    valores = []
    for corrida in corridas:
        km = pred[(corrida, "4 vistas")]
        valores.append(float(np.median(km)) if clave == "km_mediana" else float((5000 * np.exp(-km / 1492.7)).mean()))
    barras = eje.barh(nombres, valores, color=[FINALES[c][1] for c in corridas], height=0.62)
    for b, v in zip(barras, valores):
        eje.text(b.get_width() + tope * 0.015, b.get_y() + b.get_height() / 2,
                 formato.format(round(v)).replace(",", "."), va="center", fontsize=17, color=TINTA)
    eje.set_xlim(0, tope)
    fig.suptitle(titulo, fontsize=16, color=TINTA)
    eje.tick_params(axis="y", labelsize=17, length=0, labelcolor=TINTA)
    eje.xaxis.set_visible(False)
    eje.grid(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_visible(False)
    guardar_fijo(fig, archivo)

# Ejemplos: 30 panoramas al azar de test, real -> predicción del ganador (4 vistas).
rng = np.random.default_rng(7)
muestra = rng.choice(n_pano, 30, replace=False)
suma = np.zeros((n_pano, lg_ce.shape[1]))
np.add.at(suma, pano, lg_ce)
k = suma.argmax(1)
fig, eje = plt.subplots(figsize=(11, 5.2))
mapa_base(eje)
km_m = pred[("baseline_celdas_2m", "4 vistas")]
for p in muestra:
    i = primera[p]
    color = "#1baf7a" if km_m[p] < 750 else ("#eda100" if km_m[p] < 2500 else PIN_MODELO)
    eje.plot([test.longitude[i], proto[k[p], 1]], [test.latitude[i], proto[k[p], 0]], color=color, lw=2.2)
    eje.scatter([test.longitude[i]], [test.latitude[i]], s=40, color=PIN_REAL, zorder=5, edgecolor="white",
                linewidth=0.8)
from matplotlib.lines import Line2D  # noqa: E402

eje.legend(handles=[Line2D([], [], color="#1baf7a", lw=2, label="< 750 km"),
                    Line2D([], [], color="#eda100", lw=2, label="750 – 2500 km"),
                    Line2D([], [], color=PIN_MODELO, lw=2, label="> 2500 km"),
                    Line2D([], [], marker="o", ls="", color=PIN_REAL, label="ubicación real")],
           loc="lower left", fontsize=16)
guardar(fig, "test_ejemplos.png")

# Mapa: error mediano por país del ganador (4 vistas).
paises_test = test.country_code.to_numpy()[primera]
df = pd.DataFrame({"pais": paises_test, "km": km_m}).groupby("pais").km.agg(["median", "size"])
df = df[df["size"] >= 5]
fig, eje = plt.subplots(figsize=(11, 5.2))
m = MUNDO.merge(df, left_on="ISO_A2_EH", right_index=True, how="left")
m.plot(ax=eje, color=TIERRA, edgecolor=BORDE, linewidth=0.4)
m.dropna(subset=["median"]).plot(ax=eje, column="median", cmap=SECUENCIAL.reversed(), vmin=0, vmax=1500,
                                 edgecolor=BORDE, linewidth=0.4, legend=True,
                                 legend_kwds={"shrink": 0.6, "label": "km mediana (4 vistas)", "pad": 0.01})
eje.set_xlim(-170, 180)
eje.set_ylim(-58, 80)
eje.set_axis_off()
guardar(fig, "test_mapa_km_pais.png")

# Tablas de test (del notebook de comparación, recalculadas acá para no depender de él).
resumen = {}
for corrida, (nombre, _) in MODELOS.items():
    resumen[nombre] = {}
    for modo in ["1 vista", "4 vistas"]:
        km = pred[(corrida, modo)]
        resumen[nombre][modo] = {"km_mediana": round(float(np.median(km))),
                                 "puntaje": round(float((5000 * np.exp(-km / 1492.7)).mean())),
                                 "pct_750": round(float((km <= 750).mean() * 100), 1),
                                 "pct_2500": round(float((km <= 2500).mean() * 100), 1)}
datos_graficos["test"] = resumen

(SALIDA / "datos_graficos.json").write_text(json.dumps(datos_graficos, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in datos_graficos.items() if k != "overfitting"}, indent=1, ensure_ascii=False))
