"""Logits de las corridas de 2M sobre el split de test (shards 385-392, nunca usado en entrenamiento ni selección).

Cada imagen se decodifica una sola vez y pasa por todos los modelos. Antes de la transformación de validación se
aplica la misma recompresión que tuvieron train y val en Colab (lado menor 384 px, bicúbica, JPEG calidad 90),
para que test llegue a la red igual que llegaron las imágenes con las que se entrenó y validó.

Escribe outputs/entrenamiento_2m/test_filas.parquet (archivo, fila, panoid, lat, lon, país, celda real; en el
orden de los logits) y <corrida>/logits/<checkpoint>__test.npy para cada corrida.

Uso: .venv/Scripts/python.exe scripts/inferir_test.py [--corridas a b] [--datos ../world-streetview-500k/data]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entrenamiento.config import Config  # noqa: E402
from src.entrenamiento.datos import leer_filas, tareas_por_archivo, transformaciones  # noqa: E402
from src.entrenamiento.entrenamiento import construir_modelo, resolver  # noqa: E402

BASE = Path("outputs/entrenamiento_2m")


def recomprimir(im: Image.Image, lado: int = 384, calidad: int = 90) -> Image.Image:
    """Lo mismo que scripts/recomprimir_shards.py, en memoria."""
    ancho, alto = im.size
    escala = lado / min(ancho, alto)
    if escala >= 1:
        return im
    im = im.resize((round(ancho * escala), round(alto * escala)), Image.BICUBIC)
    salida = io.BytesIO()
    im.save(salida, "JPEG", quality=calidad)
    return Image.open(io.BytesIO(salida.getvalue())).convert("RGB")


class Imagenes(IterableDataset):
    def __init__(self, tareas, carpeta: Path, transform):
        self.tareas, self.carpeta, self.transform = tareas, carpeta, transform

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for archivo, filas, ids in self.tareas[wid::nw]:
            for datos, i in leer_filas(self.carpeta / archivo, filas, ids):
                try:
                    with Image.open(io.BytesIO(datos)) as im:
                        yield self.transform(recomprimir(im.convert("RGB"))), i, True
                except Exception:  # noqa: BLE001
                    yield torch.zeros(3, 224, 224), i, False


def checkpoint_de(carpeta: Path) -> tuple[str, Path]:
    """completo_2m: su último snapshot corto (el mejor en val); el resto: mejor.pt."""
    snaps = sorted((carpeta / "snapshots").glob("snapshot_corto_*.pt"))
    return (snaps[-1].stem, snaps[-1]) if snaps else ("mejor", carpeta / "mejor.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridas", nargs="*")
    ap.add_argument("--datos", default="../world-streetview-500k/data")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    raiz = resolver(BASE)
    corridas = args.corridas or sorted(p.name for p in raiz.iterdir()
                                       if (p / "config.json").exists() and p.name != "prueba_200k")

    cfg0 = Config(carpeta_corrida="x")
    splits = pq.read_table(resolver(cfg0.splits), columns=["split", "archivo", "fila", "panoid", "ind_cluster",
                                                           "country_code"]).to_pandas()
    coords = pq.read_table(resolver(cfg0.manifest), columns=["archivo", "fila", "latitude", "longitude"]).to_pandas()
    test = (splits[splits.split == "test"].merge(coords, on=["archivo", "fila"], how="left", validate="one_to_one")
            .sort_values(["archivo", "fila"]).reset_index(drop=True))
    test["country_code"] = test["country_code"].astype(str).str.strip().str.upper()
    test.drop(columns="split").to_parquet(raiz / "test_filas.parquet", index=False)
    print(f"test: {len(test)} imágenes, {test.panoid.nunique()} panoramas, {test.archivo.nunique()} archivos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = {}
    for nombre in corridas:
        carpeta = raiz / nombre
        cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        ck, ruta = checkpoint_de(carpeta)
        estado = torch.load(ruta, map_location="cpu", weights_only=False)["modelo"]
        num_clases = estado["fc.1.weight"].shape[0]
        m = construir_modelo(num_clases, cfg["dropout"], preentrenado=False)
        m.load_state_dict(estado)
        modelos[nombre] = (ck, m.to(device).eval())
        print(f"  {nombre}: {ck} ({num_clases} clases)")

    _, transform_val = transformaciones(224)
    ids = np.arange(len(test))
    flujo = Imagenes(tareas_por_archivo(test.archivo.to_numpy(), test.fila.to_numpy(), ids),
                     resolver(args.datos), transform_val)
    loader = DataLoader(flujo, batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda")
    salidas = {n: np.full((len(test), m.fc[1].out_features), np.nan, np.float32) for n, (_, m) in modelos.items()}
    fallidas, hechas, t0 = 0, 0, time.time()
    with torch.inference_mode():
        for x, i, ok in loader:
            x = x.to(device, non_blocking=True)
            for n, (_, m) in modelos.items():
                salidas[n][i.numpy()] = m(x).float().cpu().numpy()
            fallidas += int((~ok).sum())
            hechas += len(i)
            if hechas % (args.batch * 40) < args.batch:
                print(f"  {hechas}/{len(test)} ({hechas / (time.time() - t0):.0f} img/s)", flush=True)
    for n, (ck, _) in modelos.items():
        if np.isnan(salidas[n]).any():
            raise RuntimeError(f"Faltan predicciones para {n}")
        np.save(raiz / n / "logits" / f"{ck}__test.npy", salidas[n])
    print(f"Listo en {(time.time() - t0) / 60:.1f} min ({fallidas} imágenes no decodificaron)")


if __name__ == "__main__":
    main()
