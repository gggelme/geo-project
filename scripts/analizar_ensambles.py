"""Ensambles de snapshots a partir de los logits guardados por la evaluación final de una corrida.

Uso: .venv/Scripts/python.exe scripts/analizar_ensambles.py --corrida outputs/entrenamiento_2m/prueba_200k
Escribe <corrida>/ensambles.csv (métricas por ensamble y tabla) y ensambles_bootstrap.csv (diferencias contra el
modelo base, con intervalo de confianza del 95% por bootstrap pareado sobre panoramas).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entrenamiento.config import Config  # noqa: E402
from src.entrenamiento.entrenamiento import clases_y_tablas, haversine_np, metricas, resolver, softmax_np  # noqa: E402

N_BOOTSTRAP = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrida", required=True)
    args = ap.parse_args()
    carpeta = resolver(args.corrida)
    datos = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    datos.pop("huella", None)
    # Corridas de Colab: los insumos quedaron con rutas de la máquina (/content/...); se usan los locales por defecto.
    for campo in ("manifest", "splits", "val_global"):
        if datos.get(campo) and not resolver(datos[campo]).exists():
            datos.pop(campo)
    cfg = Config(**datos)
    proto, tablas, _ = clases_y_tablas(cfg)  # Celdas o países, igual que en el entrenamiento.
    dist_proto = haversine_np(proto[:, None, 0], proto[:, None, 1], proto[None, :, 0], proto[None, :, 1])

    modelos = sorted({p.name.split("__")[0] for p in (carpeta / "logits").glob("*.npy")})
    cortos = [m for m in modelos if m.startswith("snapshot_corto")]
    largos = [m for m in modelos if m.startswith("snapshot_largo")]
    ensambles = {"base": ["mejor"]}
    for nombre, grupo in (("corto", cortos), ("largo", largos)):
        if grupo:
            ensambles[f"ultimo {nombre} (1)"] = grupo[-1:]
            ensambles[f"{nombre} ({len(grupo)})"] = grupo
            ensambles[f"base+{nombre} ({len(grupo) + 1})"] = ["mejor"] + grupo
    if cortos and largos:  # Solo si la corrida tuvo las dos variantes.
        ensambles[f"corto+largo ({len(cortos) + len(largos)})"] = cortos + largos
        ensambles[f"base+corto+largo ({len(cortos) + len(largos) + 1})"] = ["mejor"] + cortos + largos
        ensambles["ultimo corto + ultimo largo (2)"] = [cortos[-1], largos[-1]]

    filas, boot = [], []
    rng = np.random.default_rng(0)
    for tabla in ["val", "val_global"]:
        t = tablas[tabla]
        dist = haversine_np(t.lat[:, None], t.lon[:, None], proto[None, :, 0], proto[None, :, 1])
        idx = np.arange(len(t))
        probs = {m: softmax_np(np.load(carpeta / "logits" / f"{m}__{tabla}.npy").astype(np.float64)) for m in modelos}
        # Diversidad: fracción de imágenes donde dos miembros predicen distinta celda (promedio sobre pares).
        _, pano = np.unique(t.panoid, return_inverse=True)
        n_pano = pano.max() + 1
        aciertos, kms = {}, {}
        for nombre, miembros in ensambles.items():
            p = np.mean([probs[m] for m in miembros], axis=0)
            met = metricas(np.log(np.clip(p, 1e-12, None)), t.celda, dist, dist_proto, cfg.tau_km)
            preds = [probs[m].argmax(1) for m in miembros]
            desacuerdo = (np.mean([(a != b).mean() for i, a in enumerate(preds) for b in preds[i + 1:]])
                          if len(preds) > 1 else 0.0)
            filas.append({"tabla": tabla, "ensamble": nombre, "miembros": len(miembros), **met,
                          "desacuerdo_top1": desacuerdo})
            pred = p.argmax(1)
            aciertos[nombre], kms[nombre] = (pred == t.celda).astype(float), dist[idx, pred]
        # Bootstrap pareado por panorama (las vistas de un panorama no son independientes).
        muestras = rng.integers(0, n_pano, size=(N_BOOTSTRAP, n_pano))
        pesos = np.stack([np.bincount(pano_b, minlength=n_pano)[pano] for pano_b in muestras])  # (B, n_imagenes)
        for nombre in ensambles:
            if nombre == "base":
                continue
            d_top1 = (pesos @ (aciertos[nombre] - aciertos["base"])) / pesos.sum(1)
            d_km = (pesos @ (kms[nombre] - kms["base"])) / pesos.sum(1)
            boot.append({"tabla": tabla, "ensamble": nombre,
                         "dif_top1": float((aciertos[nombre] - aciertos["base"]).mean()),
                         "dif_top1_ic95": f"[{np.percentile(d_top1, 2.5):+.4f}, {np.percentile(d_top1, 97.5):+.4f}]",
                         "dif_km_media": float((kms[nombre] - kms["base"]).mean()),
                         "dif_km_media_ic95": f"[{np.percentile(d_km, 2.5):+.1f}, {np.percentile(d_km, 97.5):+.1f}]"})

    res = pd.DataFrame(filas)
    res.to_csv(carpeta / "ensambles.csv", index=False)
    pd.DataFrame(boot).to_csv(carpeta / "ensambles_bootstrap.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["tabla", "ensamble", "ce_haversine", "top1", "top5", "km_mediana", "km_media", "km_mediana_mbr", "ece",
               "desacuerdo_top1"]].round(4).to_string(index=False))
    print()
    print(pd.DataFrame(boot).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
