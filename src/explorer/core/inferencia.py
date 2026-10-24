"""Pasada completa del modelo sobre un parquet, con caché en disco."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .datos import leer_imagen
from .modelo import cargar_modelo, capa_features
from .paths import raiz_proyecto
import pyarrow.parquet as pq


class _Imagenes(Dataset):
    def __init__(self, ruta, n, transform):
        self.ruta, self.n, self.transform = ruta, n, transform

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.transform(leer_imagen(self.ruta, i))


def _clave(ruta_pt, ruta_parquet):
    h = hashlib.sha256()
    for p in (ruta_pt, ruta_parquet):
        st = p.stat()
        h.update(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def predecir(ruta_pt, ruta_parquet, device="cpu", batch_size=64, progreso=None):
    """Devuelve dict con probs (N,C), features (N,D) float16 y y_pred (N,)."""
    ruta_pt, ruta_parquet = Path(ruta_pt), Path(ruta_parquet)
    cache = raiz_proyecto() / ".cache" / "explorer" / f"{_clave(ruta_pt, ruta_parquet)}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}

    modelo, config, transform = cargar_modelo(ruta_pt, device)
    n = pq.ParquetFile(ruta_parquet).metadata.num_rows
    loader = DataLoader(_Imagenes(ruta_parquet, n, transform), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    capturado = {}
    gancho = capa_features(modelo, config).register_forward_hook(
        lambda m, i, o: capturado.__setitem__("f", o.flatten(1)))
    probs, feats = [], []
    try:
        with torch.inference_mode():
            for k, x in enumerate(tqdm(loader, desc="Inferencia")):
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    logits = modelo(x.to(device))
                probs.append(torch.softmax(logits.float(), 1).cpu().numpy().astype(np.float16))
                feats.append(capturado["f"].float().cpu().numpy().astype(np.float16))
                if progreso:
                    progreso((k + 1) / len(loader))
    finally:
        gancho.remove()
    res = {"probs": np.concatenate(probs), "features": np.concatenate(feats)}
    res["y_pred"] = res["probs"].astype(np.float32).argmax(1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporal = cache.with_name(cache.stem + ".tmp.npz")
    np.savez(temporal, **res)
    temporal.replace(cache)
    return res
