"""Manifest del dataset completo (train de 393 parquets + val oficial de HF).

Una fila por imagen con: ubicación en el shard (para leerla después sin recorrer todo), metadatos, celda K=70,
pertenencia a la validación global actual y un chequeo de cada imagen (decodifica, tamaño, estadísticas de píxeles,
md5 de los bytes). El manifest es solo un inventario: los splits train/val/test se arman después, en otro paso.

Etapa 1 (lenta, en paralelo, retomable): un archivo `partes/<shard>.parquet` por shard. Si el proceso se corta,
al volver a ejecutarlo se saltean los shards ya terminados (escritura atómica: tmp + os.replace).
Etapa 2 (rápida): junta las partes, calcula celdas y banderas y escribe `manifest.parquet` + `manifest_config.json`.

Uso (desde la raíz del proyecto):
    .venv/Scripts/python.exe scripts/construir_manifest.py --limite-archivos 2   # prueba
    .venv/Scripts/python.exe scripts/construir_manifest.py                       # todo
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import warnings
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parents[1]
COLUMNAS_META = ["image_id", "panoid", "country_code", "date", "latitude", "longitude", "elevation"]
LADO_MINIMO = 64  # Menos de esto en algún lado: imagen inutilizable para un crop de 224.
STD_PLACEHOLDER = 3.0  # Desvío de luminancia (0-255) por debajo del cual la imagen es casi uniforme.
# Fallas del sensor de Street View: imagen teñida de magenta (rojo y azul muy por encima del verde).
UMBRAL_DIF_MAGENTA = 30  # Píxel magenta: R - G > 30 y B - G > 30.
FRAC_MAGENTA = 0.9  # Defectuosa si más del 90% de sus píxeles son magenta (revisado a ojo por franjas).

VERSION = 2  # Cambiarla invalida las partes ya calculadas (si cambia lo que se mide por imagen).


def resolver(ruta: str | Path) -> Path:
    p = Path(ruta).expanduser()
    return (p if p.is_absolute() else RAIZ / p).resolve()


def leer_data_url() -> str:
    if os.environ.get("DATA_URL"):
        return os.environ["DATA_URL"]
    env = RAIZ / ".env"
    if env.is_file():
        for linea in env.read_text(encoding="utf-8-sig").splitlines():
            clave, sep, valor = linea.partition("=")
            if sep and clave.strip() == "DATA_URL":
                return valor.strip().strip("\"'")
    raise ValueError("Configurá DATA_URL en el .env o pasá --datos.")


# ---------------------------------------------------------------- etapa 1: un shard por proceso
def revisar_imagen(contenido: dict | None) -> dict:
    """Chequea una imagen sin transformarla. Nunca lanza: devuelve el error como texto."""
    r = {"bytes_imagen": 0, "md5": None, "ancho": -1, "alto": -1, "modo": None, "formato": None,
         "luminancia_media": np.nan, "luminancia_std": np.nan, "media_r": np.nan, "media_g": np.nan,
         "media_b": np.nan, "frac_magenta": np.nan, "error": None}
    datos = None if contenido is None else contenido.get("bytes")
    if not datos:
        r["error"] = "sin_bytes" if contenido is None or not contenido.get("path") else "solo_path"
        return r
    r["bytes_imagen"] = len(datos)
    r["md5"] = hashlib.md5(datos).hexdigest()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(datos)) as im:
                r["formato"], r["modo"] = im.format, im.mode
                r["ancho"], r["alto"] = im.size
                im.load()  # Decodificación completa: acá salta un JPEG truncado o dañado.
                rgb = im.convert("RGB")
                rgb = rgb.reduce(max(1, min(rgb.size) // 64))
                px = np.asarray(rgb.convert("L"), dtype=np.float32)
                r["luminancia_media"], r["luminancia_std"] = float(px.mean()), float(px.std())
                c = np.asarray(rgb, dtype=np.float32)
                R, G, B = c[..., 0], c[..., 1], c[..., 2]
                r["media_r"], r["media_g"], r["media_b"] = float(R.mean()), float(G.mean()), float(B.mean())
                r["frac_magenta"] = float(((R - G > UMBRAL_DIF_MAGENTA) & (B - G > UMBRAL_DIF_MAGENTA)).mean())
    except Exception as e:  # noqa: BLE001 — se registra cualquier falla de decodificación.
        r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return r


def procesar_shard(args: tuple[str, str, str, int]) -> tuple[str, int, float]:
    archivo, nombre_relativo, carpeta_partes, tamano_lote = args
    destino = Path(carpeta_partes) / (nombre_relativo.replace("/", "__") + ".parquet")
    inicio = time.perf_counter()
    with pq.ParquetFile(archivo) as pf:
        n_filas = pf.metadata.num_rows
        if destino.is_file():
            with pq.ParquetFile(destino) as hecho:  # Cerrarlo: en Windows un archivo abierto no se puede reemplazar.
                completo = (hecho.metadata.num_rows == n_filas
                            and hecho.schema_arrow.metadata.get(b"version") == str(VERSION).encode())
            if completo:
                return nombre_relativo, n_filas, 0.0
        meta = pf.read(columns=COLUMNAS_META).to_pandas()
        grupos = np.concatenate([np.full(pf.metadata.row_group(g).num_rows, g) for g in range(pf.num_row_groups)])
        revisiones = []
        for lote in pf.iter_batches(batch_size=tamano_lote, columns=["image"]):
            revisiones.extend(revisar_imagen(c) for c in lote.column("image").to_pylist())
    if len(revisiones) != n_filas or len(meta) != n_filas:
        raise RuntimeError(f"{nombre_relativo}: filas leídas {len(revisiones)} / {len(meta)}, esperadas {n_filas}")
    tabla = pd.concat([meta, pd.DataFrame(revisiones)], axis=1)
    tabla.insert(0, "archivo", nombre_relativo)
    tabla.insert(1, "fila", np.arange(n_filas, dtype=np.int32))
    tabla.insert(2, "row_group", grupos.astype(np.int16))
    t = pa.Table.from_pandas(tabla, preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"version": str(VERSION).encode()})
    tmp = destino.with_suffix(".tmp")
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, destino)
    return nombre_relativo, n_filas, time.perf_counter() - inicio


# ---------------------------------------------------------------- etapa 2: consolidar
def to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def completar_country_code(tabla: pd.DataFrame, ruta_paises: Path) -> pd.Series:
    # Igual que en los clasificadores: país del polígono que contiene el punto (o el más cercano).
    codigos = tabla["country_code"].astype("string").str.strip().str.upper().replace("", pd.NA)
    faltan = codigos.isna() & tabla["coords_validas"]
    if not faltan.any():
        return codigos
    import geopandas as gpd

    paises = gpd.read_file(ruta_paises).to_crs("EPSG:4326")
    paises = paises.loc[paises["ISO_A2_EH"] != "-99", ["ISO_A2_EH", "geometry"]]
    puntos = gpd.GeoDataFrame({"fila": tabla.index[faltan]},
                              geometry=gpd.points_from_xy(tabla.loc[faltan, "longitude"], tabla.loc[faltan, "latitude"]),
                              crs="EPSG:4326")
    dentro = puntos.sjoin(paises, how="left", predicate="within").drop_duplicates("fila").set_index("fila")["ISO_A2_EH"]
    sin_pais = dentro.index[dentro.isna()]
    if len(sin_pais):
        cerca = (puntos[puntos["fila"].isin(sin_pais)].to_crs("EPSG:3857")
                 .sjoin_nearest(paises.to_crs("EPSG:3857"), how="left").drop_duplicates("fila"))
        dentro.loc[cerca["fila"].to_numpy()] = cerca["ISO_A2_EH"].to_numpy()
    codigos.loc[dentro.index] = dentro.to_numpy()
    return codigos


def huella_ids(df: pd.DataFrame) -> str:
    pares = df[["image_id", "panoid"]].sort_values(["image_id", "panoid"]).to_numpy().tolist()
    return hashlib.sha256(json.dumps(pares, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def consolidar(args, archivos: list[tuple[Path, str]], carpeta_partes: Path, inventario: list[dict]) -> None:
    partes = [carpeta_partes / (rel.replace("/", "__") + ".parquet") for _, rel in archivos]
    m = pd.concat([pq.read_table(p).to_pandas() for p in partes], ignore_index=True)
    m.insert(0, "split_origen", np.where(m["archivo"].str.contains("val-"), "val_hf", "train_hf"))
    for c in ("image_id", "panoid"):
        m[c] = m[c].astype("string").str.strip().replace("", pd.NA)

    # Coordenadas, país y celda.
    lat, lon = m["latitude"].to_numpy(float), m["longitude"].to_numpy(float)
    m["coords_validas"] = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    m["country_code_original"] = m["country_code"]
    m["country_code"] = completar_country_code(m, resolver(args.paises))

    ruta_proto = resolver(args.prototipos)
    prototipos = pd.read_csv(ruta_proto).sort_values("ind_cluster").reset_index(drop=True)
    assert (prototipos["ind_cluster"].to_numpy() == np.arange(len(prototipos))).all()
    centroides = to_xyz(prototipos["lat"].to_numpy(float), prototipos["lon"].to_numpy(float))
    celda = np.full(len(m), -1, dtype=np.int16)
    ok = m["coords_validas"].to_numpy()
    for i in range(0, len(m), 200_000):  # Por bloques: (2M x 70) de una vez es mucha RAM.
        sl = slice(i, i + 200_000)
        sub = ok[sl]
        celda[sl][sub] = (to_xyz(lat[sl][sub], lon[sl][sub]) @ centroides.T).argmax(axis=1)
    m["ind_cluster"] = celda

    # Validación global actual (la de los experimentos con 20k): se excluye por panoid e image_id.
    val = pq.read_table(resolver(args.validacion_ids), columns=["image_id", "panoid"]).to_pandas()
    m["en_validacion_global"] = m["image_id"].isin(set(val["image_id"])) | m["panoid"].isin(set(val["panoid"]))

    # Consistencia por panorama (solo train_hf: en val_hf cada panorama trae una sola vista).
    tr = m["split_origen"] == "train_hf"
    g = m.loc[tr & m["panoid"].notna()].groupby("panoid")
    por_pano = pd.DataFrame({"vistas": g.size(), "paises": g["country_code"].nunique(),
                             "dlat": g["latitude"].max() - g["latitude"].min(),
                             "dlon": g["longitude"].max() - g["longitude"].min(),
                             "celdas": g["ind_cluster"].nunique()})
    m["vistas_panorama"] = m["panoid"].map(por_pano["vistas"]).fillna(1).astype(np.int8)

    # Duplicados exactos de bytes entre filas distintas.
    con_md5 = m["md5"].notna()
    m["md5_repetido"] = False
    m.loc[con_md5, "md5_repetido"] = m.loc[con_md5, "md5"].duplicated(keep=False)

    # Defectuosas (blancas sobreexpuestas, negras, teñidas de magenta por fallas del sensor con poca luz).
    # Umbrales elegidos revisando las imágenes a ojo (2026-11-13). Magenta: > 0.9 no tiene contenido; entre 0.5 y
    # 0.9 son sobre todo túneles teñidos donde la escena se reconoce, y por debajo hay muchas fotos válidas.
    # El criterio por color medio (R y B sobre G) se probó y se abandonó: marca tierra rojiza con cielo azul.
    m["sospechosa_uniforme"] = m["luminancia_std"] < STD_PLACEHOLDER
    m["sospechosa_magenta"] = m["frac_magenta"] > FRAC_MAGENTA

    # Motivo de descarte (vacío = usable).
    motivo = pd.Series("", index=m.index, dtype=object)
    reglas = [
        (m["image_id"].isna() | m["panoid"].isna(), "ids_invalidos"),
        (~m["coords_validas"], "coords_invalidas"),
        (m["error"].notna(), "no_decodifica"),
        ((m["ancho"] < LADO_MINIMO) | (m["alto"] < LADO_MINIMO), "muy_chica"),
        (m["country_code"].isna(), "sin_pais"),
        # Las 89 con luminancia_std < 3 se revisaron a ojo (2026-11-13): blancas sobreexpuestas, negras sin
        # contenido y magentas puras. Se descartan todas.
        (m["sospechosa_uniforme"], "uniforme"),
        (m["sospechosa_magenta"], "magenta"),
    ]
    for mascara, nombre in reglas:
        motivo[mascara.to_numpy() & (motivo == "").to_numpy()] = nombre
    m["motivo_descarte"] = motivo
    m["imagen_ok"] = m["motivo_descarte"] == ""

    # Fuga entre el val oficial de HF y train.
    val_hf, train_hf = m[m["split_origen"] == "val_hf"], m[tr]
    fuga_val_hf = {
        "panoids_en_train": int(val_hf["panoid"].isin(set(train_hf["panoid"])).sum()),
        "image_ids_en_train": int(val_hf["image_id"].isin(set(train_hf["image_id"])).sum()),
        "md5_en_train": int(val_hf["md5"].isin(set(train_hf["md5"].dropna())).sum()),
    }

    salida = resolver(args.salida)
    columnas = ["split_origen", "archivo", "fila", "row_group", "image_id", "panoid", "country_code",
                "country_code_original", "date", "latitude", "longitude", "elevation", "coords_validas",
                "ind_cluster", "en_validacion_global", "vistas_panorama", "bytes_imagen", "md5", "md5_repetido",
                "ancho", "alto", "modo", "formato", "luminancia_media", "luminancia_std", "media_r", "media_g",
                "media_b", "frac_magenta", "error", "sospechosa_uniforme", "sospechosa_magenta", "motivo_descarte",
                "imagen_ok"]
    tmp = salida / "manifest.parquet.tmp"
    pq.write_table(pa.Table.from_pandas(m[columnas], preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, salida / "manifest.parquet")

    ok_train = m[tr & m["imagen_ok"] & ~m["en_validacion_global"]]
    resumen = {
        "filas": int(len(m)), "por_split_origen": m["split_origen"].value_counts().to_dict(),
        "descartes": m.loc[~m["imagen_ok"], "motivo_descarte"].value_counts().to_dict(),
        "errores_decodificacion": m["error"].dropna().str.split(":").str[0].value_counts().head(20).to_dict(),
        "sospechosas_uniformes": int(m["sospechosa_uniforme"].sum()),
        "sospechosas_magenta": int(m["sospechosa_magenta"].sum()),
        "filas_md5_repetido": int(m["md5_repetido"].sum()),
        "country_code_completado": int((m["country_code_original"].astype("string").str.strip().fillna("") == "").sum()),
        "en_validacion_global": int(m["en_validacion_global"].sum()),
        "train_usable_fuera_de_validacion_global": {"imagenes": int(len(ok_train)),
                                                     "panoramas": int(ok_train["panoid"].nunique())},
        "panoramas_train": {"total": int(len(por_pano)),
                            "vistas": por_pano["vistas"].value_counts().sort_index().to_dict(),
                            "con_varios_paises": int((por_pano["paises"] > 1).sum()),
                            "con_varias_celdas": int((por_pano["celdas"] > 1).sum()),
                            "max_dif_lat": float(por_pano["dlat"].max()), "max_dif_lon": float(por_pano["dlon"].max())},
        "tamanos_frecuentes": {f"{a}x{h}": int(n) for (a, h), n in
                               m.groupby(["ancho", "alto"]).size().sort_values(ascending=False).head(5).items()},
        "bytes_por_imagen": {"media": float(m["bytes_imagen"].mean()), "p01": float(m["bytes_imagen"].quantile(0.01))},
        "fuga_val_hf_vs_train": fuga_val_hf,
        "imagenes_por_celda_train_usable": ok_train["ind_cluster"].value_counts().sort_index().to_dict(),
    }
    config = {
        "tipo": "manifest_dataset_completo", "version": VERSION,
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/construir_manifest.py",
        "datos": str(resolver(args.datos)),
        "prototipos": {"ruta": str(ruta_proto), "sha256": hashlib.sha256(ruta_proto.read_bytes()).hexdigest()},
        "validacion_global": {"ruta": str(resolver(args.validacion_ids)), "sha256_ids": huella_ids(val)},
        "umbrales": {"lado_minimo": LADO_MINIMO, "std_placeholder": STD_PLACEHOLDER,
                     "umbral_dif_magenta": UMBRAL_DIF_MAGENTA, "frac_magenta": FRAC_MAGENTA},
        "archivos_fuente": inventario,
        "resumen": resumen,
    }
    (salida / "manifest_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str),
                                                 encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2, default=str)[:6000])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=None, help="Carpeta con train-*.parquet y val-*.parquet (default: DATA_URL)")
    ap.add_argument("--salida", default="datasets/manifest_2m")
    ap.add_argument("--prototipos", default="src/notebooks/preprocessing/assets/prototipos_train-completo_k70.csv")
    ap.add_argument("--validacion-ids", default="datasets/samples/data/validation/validation_ids.parquet")
    ap.add_argument("--paises", default="src/notebooks/eda/assets/ne_110m_admin_0_countries.geojson")
    ap.add_argument("--procesos", type=int, default=3, help="Procesos de la etapa 1 (cada uno usa ~1 núcleo)")
    ap.add_argument("--tamano-lote", type=int, default=64, help="Imágenes leídas por vez (limita la RAM)")
    ap.add_argument("--limite-archivos", type=int, default=None, help="Solo los primeros N shards de train (prueba)")
    args = ap.parse_args()
    args.datos = args.datos or leer_data_url()

    ruta_datos = resolver(args.datos)
    train = sorted(ruta_datos.glob("**/train-*.parquet"))
    val = sorted(ruta_datos.glob("**/val-*.parquet"))
    if not train:
        raise FileNotFoundError(f"No hay train-*.parquet en {ruta_datos}")
    if args.limite_archivos:
        train = train[: args.limite_archivos]
    archivos = [(p, p.relative_to(ruta_datos).as_posix()) for p in (*train, *val)]

    salida = resolver(args.salida)
    carpeta_partes = salida / "partes"
    carpeta_partes.mkdir(parents=True, exist_ok=True)
    inventario = [{"archivo": rel, "filas": pq.ParquetFile(p).metadata.num_rows, "bytes": p.stat().st_size}
                  for p, rel in archivos]
    print(f"{len(train)} shards de train + {len(val)} de val -> {salida}")

    tareas = [(str(p), rel, str(carpeta_partes), args.tamano_lote) for p, rel in archivos]
    inicio = time.perf_counter()
    with Pool(args.procesos) as pool:
        barra = tqdm(pool.imap_unordered(procesar_shard, tareas), total=len(tareas), desc="Shards")
        for rel, n, seg in barra:
            if seg:
                barra.set_postfix_str(f"{rel}: {n / seg:.0f} img/s")
    print(f"Etapa 1: {(time.perf_counter() - inicio) / 60:.1f} min")
    consolidar(args, archivos, carpeta_partes, inventario)


if __name__ == "__main__":
    main()
