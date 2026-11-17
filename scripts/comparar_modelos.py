"""Compara corridas de 2M (celdas y países) sobre las mismas imágenes de validación, con los logits guardados.

Para cada corrida, tabla (val del split, val_global de 20k) y modo (cada vista por separado, o las 4 vistas de un
panorama combinadas promediando logits):
  - top1 en su propia tarea, km (mediana, media, % por umbral) y puntaje GeoGuessr medio (5000·exp(-d/1492.7));
  - top1 de país para todos los modelos: los de celdas suman la probabilidad de cada celda repartida entre países
    según la composición de esa celda en train. Las imágenes de países que un modelo no tiene cuentan como error.
Diferencias contra la referencia con IC 95% por bootstrap pareado sobre panoramas.

Uso: .venv/Scripts/python.exe scripts/comparar_modelos.py [--corridas a b c] [--referencia baseline_celdas_2m]
Escribe outputs/entrenamiento_2m/comparacion.csv y comparacion_bootstrap.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entrenamiento.config import Config  # noqa: E402
from src.entrenamiento.entrenamiento import clases_y_tablas, haversine_np, resolver, softmax_np  # noqa: E402

BASE = Path("outputs/entrenamiento_2m")
N_BOOTSTRAP = 1000
UMBRALES = [25, 200, 750, 2500]


def cargar_cfg(carpeta: Path) -> Config:
    datos = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    datos.pop("huella", None)
    for campo in ("manifest", "splits", "val_global"):  # Rutas de Colab -> locales por defecto.
        if datos.get(campo) and not resolver(datos[campo]).exists():
            datos.pop(campo)
    return Config(**datos)


def codigo(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.strip().str.upper().to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridas", nargs="*")
    ap.add_argument("--referencia", default="baseline_celdas_2m")
    ap.add_argument("--checkpoint", default="mejor", help="Logits a usar; completo_2m usa su último snapshot corto")
    args = ap.parse_args()
    raiz = resolver(BASE)
    corridas = args.corridas or sorted(p.name for p in raiz.iterdir()
                                       if (p / "logits").is_dir() and p.name != "prueba_200k")

    # País real de cada imagen, en el mismo orden que las tablas de entrenamiento (val: archivo, fila).
    cfg0 = Config(carpeta_corrida="x")
    splits = pq.read_table(resolver(cfg0.splits), columns=["split", "archivo", "fila", "ind_cluster",
                                                           "country_code"]).to_pandas()
    pais_val = codigo(splits[splits.split == "val"].sort_values(["archivo", "fila"])["country_code"])
    pais_vg = codigo(pq.read_table(resolver(cfg0.val_global), columns=["country_code"]).to_pandas()["country_code"])
    tr = splits[splits.split == "train"]
    todos = sorted(set(codigo(tr["country_code"])) | set(pais_val) | set(pais_vg))
    ip = {p: i for i, p in enumerate(todos)}
    # P(país | celda) en train.
    conteo = np.zeros((tr["ind_cluster"].max() + 1, len(todos)))
    np.add.at(conteo, (tr["ind_cluster"].to_numpy(), [ip[p] for p in codigo(tr["country_code"])]), 1)
    p_pais_celda = conteo / conteo.sum(1, keepdims=True)
    pais_real = {"val": np.array([ip[p] for p in pais_val]), "val_global": np.array([ip[p] for p in pais_vg])}

    filas, por_imagen = [], {}
    for nombre in corridas:
        carpeta = raiz / nombre
        cfg = cargar_cfg(carpeta)
        proto, tablas, nombres = clases_y_tablas(cfg)
        ck = "snapshot_corto_4" if (carpeta / "logits" / "snapshot_corto_4__val.npy").exists() else args.checkpoint
        if cfg.tarea == "paises":
            a_pais = np.zeros((len(nombres), len(todos)))
            a_pais[np.arange(len(nombres)), [ip[p] for p in nombres]] = 1
        else:
            a_pais = p_pais_celda
        for tabla in ["val", "val_global"]:
            t = tablas[tabla]
            assert len(t) == len(pais_real[tabla])
            logits = np.load(carpeta / "logits" / f"{ck}__{tabla}.npy").astype(np.float64)
            _, pano = np.unique(t.panoid, return_inverse=True)
            for modo in ["1 vista", "4 vistas"]:
                if modo == "4 vistas":  # Promedio de logits del panorama; la misma predicción para sus vistas.
                    suma = np.zeros((pano.max() + 1, logits.shape[1]))
                    np.add.at(suma, pano, logits)
                    lg = (suma / np.bincount(pano)[:, None])[pano]
                else:
                    lg = logits
                p = softmax_np(lg)
                pred = p.argmax(1)
                km = haversine_np(t.lat, t.lon, proto[pred, 0], proto[pred, 1])
                acierto_pais = ((p @ a_pais).argmax(1) == pais_real[tabla]).astype(float)
                puntaje = 5000 * np.exp(-km / 1492.7)
                fila = {"corrida": nombre, "checkpoint": ck, "tarea": cfg.tarea, "clases": len(nombres),
                        "tabla": tabla, "modo": modo, "imagenes": len(t),
                        "top1_tarea": float((pred == t.celda).mean()), "top1_pais": acierto_pais.mean(),
                        "km_mediana": float(np.median(km)), "km_media": float(km.mean()),
                        "puntaje_geoguessr": float(puntaje.mean())}
                fila.update({f"pct_{u}km": float((km <= u).mean()) for u in UMBRALES})
                filas.append(fila)
                por_imagen[(nombre, tabla, modo)] = (pano, acierto_pais, km, puntaje)

    res = pd.DataFrame(filas)
    res.to_csv(raiz / "comparacion.csv", index=False)

    rng = np.random.default_rng(0)
    boot = []
    for tabla in ["val", "val_global"]:
        for modo in ["1 vista", "4 vistas"]:
            pano, *ref = por_imagen[(args.referencia, tabla, modo)]
            n = pano.max() + 1
            pesos = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)[pano] for _ in range(N_BOOTSTRAP)])
            for nombre in corridas:
                if nombre == args.referencia:
                    continue
                _, *otro = por_imagen[(nombre, tabla, modo)]
                fila = {"corrida": nombre, "vs": args.referencia, "tabla": tabla, "modo": modo}
                for metrica, a, b in zip(["top1_pais", "km_media", "puntaje_geoguessr"], otro, ref):
                    d = (pesos @ (a - b)) / pesos.sum(1)
                    fila[f"dif_{metrica}"] = float((a - b).mean())
                    fila[f"dif_{metrica}_ic95"] = f"[{np.percentile(d, 2.5):+.4g}, {np.percentile(d, 97.5):+.4g}]"
                boot.append(fila)
    pd.DataFrame(boot).to_csv(raiz / "comparacion_bootstrap.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(res.drop(columns=["checkpoint", "imagenes"]).round(4).to_string(index=False))
    print()
    print(pd.DataFrame(boot).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
