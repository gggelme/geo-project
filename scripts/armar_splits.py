"""Splits train / val / test por parquets enteros, a partir de datasets/manifest_2m/manifest.parquet.

- val y test son bloques de shards de train consecutivos (por defecto 4 y 8: ~5k y ~10k panoramas); el resto es train.
- test_hf: el val oficial de HF (val-00000-of-00001.parquet), test externo extra.
- Panoramas partidos entre dos shards: si los dos shards quedan en el mismo split no pasa nada; si no, el
  panorama queda solo en train (sus filas en el shard de val/test se excluyen; si cruza val-test, se excluye de ambos). Se verifica que un panorama
  aparezca únicamente en shards consecutivos y en sus bordes.
- De train se excluye además la validación global actual (para comparar con los modelos de 20k) y las imágenes
  con motivo_descarte. De val/test solo se excluyen las imágenes con motivo_descarte.
- Distribución: distancia de variación total (TVD) por país y por celda de cada split contra el train completo,
  comparada con la TVD de sortear al azar la misma cantidad de panoramas (referencia de "tan bueno como el azar").

Uso: .venv/Scripts/python.exe scripts/armar_splits.py [--val 381-384] [--test 385-392]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[1]


def resolver(ruta):
    p = Path(ruta).expanduser()
    return (p if p.is_absolute() else RAIZ / p).resolve()


def rango(texto: str) -> list[int]:
    a, _, b = texto.partition("-")
    return list(range(int(a), int(b or a) + 1))


def numero_shard(archivo: str) -> int:
    m = re.search(r"train-(\d+)-of-", archivo)
    return int(m.group(1)) if m else -1


def tvd(a: pd.Series, b: pd.Series) -> float:
    a, b = a / a.sum(), b / b.sum()
    return float(0.5 * a.subtract(b, fill_value=0).abs().sum())


def tvd_azar(panos: pd.DataFrame, columna: str, referencia: pd.Series, n: int, repeticiones: int, rng) -> np.ndarray:
    # TVD de sortear n panoramas al azar (contando imágenes), para comparar con los shards elegidos.
    salida = np.empty(repeticiones)
    n = min(n, len(panos))
    for i in range(repeticiones):
        sel = panos.iloc[rng.choice(len(panos), n, replace=False)]
        salida[i] = tvd(sel.groupby(columna)["vistas"].sum(), referencia)
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="datasets/manifest_2m/manifest.parquet")
    ap.add_argument("--salida", default="datasets/manifest_2m")
    ap.add_argument("--val", default="381-384", help="Shards de train para val (rango de números, inclusive)")
    ap.add_argument("--test", default="385-392", help="Shards de train para test")
    ap.add_argument("--repeticiones", type=int, default=200, help="Sorteos al azar para la referencia de TVD")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    m = pq.read_table(resolver(args.manifest), columns=[
        "split_origen", "archivo", "fila", "image_id", "panoid", "country_code", "ind_cluster",
        "en_validacion_global", "motivo_descarte", "imagen_ok"]).to_pandas()
    m["shard"] = m["archivo"].map(numero_shard)
    tr = m["split_origen"] == "train_hf"
    shards = sorted(m.loc[tr, "shard"].unique())
    val_sh, test_sh = rango(args.val), rango(args.test)
    if set(val_sh) & set(test_sh) or not set(val_sh + test_sh) <= set(shards):
        raise ValueError("val y test deben ser shards existentes y disjuntos.")

    # 1) Panoramas partidos entre shards.
    t = m[tr & m["panoid"].notna()]
    por_pano = t.groupby("panoid")["shard"].agg(["min", "max", "nunique"])
    partidos = por_pano[por_pano["nunique"] > 1]
    no_consecutivos = partidos[(partidos["nunique"] > 2) | (partidos["max"] - partidos["min"] != 1)]
    if len(no_consecutivos):
        print(f"ATENCIÓN: {len(no_consecutivos)} panoramas en shards NO consecutivos (ver splits_problemas.csv)")
    # Verificar que las filas de los partidos estén en los bordes del shard.
    filas_shard = t.groupby("shard")["fila"].max() + 1
    fil = t[t["panoid"].isin(partidos.index)]
    borde = (fil["fila"] < 4) | (fil["fila"] >= fil["shard"].map(filas_shard) - 4)
    print(f"Panoramas partidos entre shards: {len(partidos):,} "
          f"(todos en bordes: {bool(borde.all())}, consecutivos: {len(no_consecutivos) == 0})")

    # 2) Asignación.
    split_shard = {s: "train" for s in shards}
    split_shard.update({s: "val" for s in val_sh})
    split_shard.update({s: "test" for s in test_sh})
    m["split"] = np.where(tr, m["shard"].map(split_shard), "test_hf")
    # Panorama partido entre splits distintos -> queda en train.
    splits_pano = m[tr & m["panoid"].isin(partidos.index)].groupby("panoid")["split"].nunique()
    cruzados = splits_pano.index[splits_pano > 1]
    motivo = m["motivo_descarte"].fillna("").astype(object)
    motivo[(motivo == "") & m["panoid"].isin(cruzados) & (m["split"] != "train")] = "panorama_partido_entre_splits"
    motivo[(motivo == "") & (m["split"] == "train") & m["en_validacion_global"]] = "validacion_global"
    # El val oficial de HF repite algunas imágenes de train (mismo panoid y mismos bytes): fuera de test_hf.
    motivo[(motivo == "") & (m["split"] == "test_hf") & m["panoid"].isin(set(m.loc[tr, "panoid"]))] = "fuga_con_train"
    m["excluir"] = motivo
    m.loc[m["excluir"] == "", "excluir"] = pd.NA
    usable = m["excluir"].isna()

    # 3) Distribución contra el train completo (todas las imágenes de train_hf usables).
    ref = m[tr & m["imagen_ok"]]
    rng = np.random.default_rng(args.semilla)
    panos_ref = (ref.groupby("panoid").agg(country_code=("country_code", "first"),
                                           ind_cluster=("ind_cluster", "first"), vistas=("fila", "size")))
    filas_dist, informe = [], {}
    for split in ("train", "val", "test", "test_hf"):
        s = m[usable & (m["split"] == split)]
        n_panos = s["panoid"].nunique()
        info = {"imagenes": int(len(s)), "panoramas": int(n_panos),
                "paises": int(s["country_code"].nunique()), "celdas": int(s["ind_cluster"].nunique())}
        for columna in ("country_code", "ind_cluster"):
            d = tvd(s[columna].value_counts(), ref[columna].value_counts())
            info[f"tvd_{columna}"] = round(d, 4)
            if split in ("val", "test", "test_hf"):
                azar = tvd_azar(panos_ref, columna, ref[columna].value_counts(), n_panos, args.repeticiones, rng)
                info[f"tvd_{columna}_azar_p50"] = round(float(np.median(azar)), 4)
                info[f"tvd_{columna}_azar_p95"] = round(float(np.percentile(azar, 95)), 4)
                info[f"tvd_{columna}_percentil_vs_azar"] = round(float((azar < d).mean() * 100), 1)
        informe[split] = info
        for columna in ("country_code", "ind_cluster"):
            vc = s[columna].value_counts(normalize=True)
            filas_dist += [{"split": split, "variable": columna, "valor": str(k), "proporcion": float(v)}
                           for k, v in vc.items()]
    for columna in ("country_code", "ind_cluster"):
        vc = ref[columna].value_counts(normalize=True)
        filas_dist += [{"split": "referencia_train_completo", "variable": columna, "valor": str(k),
                        "proporcion": float(v)} for k, v in vc.items()]

    # 4) Salidas.
    salida = resolver(args.salida)
    excluidas = m.loc[~usable, ["split", "archivo", "fila", "image_id", "panoid", "excluir"]]
    reglas = {}
    for split in ("train", "val", "test", "test_hf"):
        archivos = sorted(m.loc[m["split"] == split, "archivo"].unique())
        exc = excluidas[excluidas["split"] == split]
        reglas[split] = {
            "archivos": archivos if split != "train" else f"{len(archivos)} shards: todos los train-*.parquet "
                                                            f"salvo {args.val} (val) y {args.test} (test)",
            "excluir_por_motivo": exc["excluir"].value_counts().to_dict(),
            # Detalle legible de las exclusiones de val/test (son pocas); las de train van al parquet.
            **({"excluir_filas": exc[["archivo", "fila", "excluir"]].to_dict("records")} if split != "train" else {}),
        }
    config = {
        "tipo": "splits_por_shard", "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/armar_splits.py", "manifest": str(resolver(args.manifest)),
        "shards_val": val_sh, "shards_test": test_sh,
        "regla_panoramas_partidos": "si cruzan splits, quedan solo en train",
        "panoramas_partidos": int(len(partidos)), "panoramas_partidos_entre_splits": int(len(cruzados)),
        "panoramas_en_shards_no_consecutivos": int(len(no_consecutivos)),
        "resumen": informe, "reglas": reglas,
    }
    (salida / "splits_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
    tabla = m.loc[usable, ["split", "archivo", "fila", "image_id", "panoid", "ind_cluster", "country_code"]]
    tmp = salida / "splits.parquet.tmp"
    pq.write_table(pa.Table.from_pandas(tabla, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, salida / "splits.parquet")
    pq.write_table(pa.Table.from_pandas(excluidas, preserve_index=False), salida / "splits_excluidas.parquet")
    pd.DataFrame(filas_dist).to_csv(salida / "splits_distribucion.csv", index=False)
    if len(no_consecutivos):
        no_consecutivos.to_csv(salida / "splits_problemas.csv")

    print(pd.DataFrame(informe).T.to_string())
    print(f"Panoramas partidos entre splits distintos (quedan en train): {len(cruzados)}")
    print("TVD: 0 = misma distribución que el train completo. 'percentil_vs_azar' < 95 => los shards elegidos "
          "están dentro de lo que daría un sorteo al azar de la misma cantidad de panoramas.")


if __name__ == "__main__":
    main()
