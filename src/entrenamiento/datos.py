"""Lectura de imágenes desde los parquets (disco local o bajados de HF) y flujo de muestras para el DataLoader.

Cada ronda es una lista de tareas (archivo, filas, ids). Los workers se reparten los archivos; de cada uno se
leen solo los row groups que tienen filas pedidas. Las imágenes pasan por un shuffle buffer de bytes JPEG
(no de tensores: 100 KB contra 600 KB por imagen) y recién al salir se decodifican y transforman.

Cada muestra sale como (tensor, id, ok). `id` es el índice global de la muestra en la ronda: con él el loop
principal sabe la celda, las coordenadas y de qué archivo vino (para marcar archivos terminados al retomar).
`ok = False` si la imagen no decodificó (el tensor es de ceros y el loop la ignora y la cuenta).
"""
from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import models, transforms


def transformaciones(tamano: int) -> tuple[transforms.Compose, transforms.Compose]:
    # Las mismas que en los notebooks de 20k (deben coincidir con config de S42).
    pesos = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    normalizar = transforms.Normalize(pesos.mean, pesos.std)
    train = transforms.Compose([
        transforms.RandomResizedCrop(tamano, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.ToTensor(), normalizar,
    ])
    val = transforms.Compose([transforms.Resize(tamano), transforms.CenterCrop(tamano), transforms.ToTensor(),
                              normalizar])
    return train, val


class Lector:
    """Devuelve la ruta local de un parquet. Con fuente HF lo baja (con reintentos) y puede borrarlo después."""

    def __init__(self, fuente: str, datos_local: str | Path, hf_repo: str = "", hf_revision: str | None = None,
                 hf_subcarpeta: str = "data", hf_cache: str | Path = "", borrar: bool = True, reintentos: int = 6):
        self.fuente, self.datos_local = fuente, Path(datos_local)
        self.hf_repo, self.hf_revision, self.hf_subcarpeta = hf_repo, hf_revision, hf_subcarpeta
        self.hf_cache, self.borrar, self.reintentos = Path(hf_cache), borrar, reintentos

    def abrir(self, archivo: str) -> Path:
        if Path(archivo).is_absolute():  # Archivos fuera del dataset (p. ej. la validación global de 20k).
            return Path(archivo)
        if self.fuente == "local":
            return self.datos_local / archivo
        from huggingface_hub import hf_hub_download

        for intento in range(self.reintentos):
            try:
                return Path(hf_hub_download(self.hf_repo, f"{self.hf_subcarpeta}/{archivo}", repo_type="dataset",
                                            revision=self.hf_revision, local_dir=str(self.hf_cache)))
            except Exception as e:  # noqa: BLE001 — red: se reintenta con espera creciente.
                if intento == self.reintentos - 1:
                    raise RuntimeError(f"No se pudo bajar {archivo} tras {self.reintentos} intentos") from e
                time.sleep(min(300, 10 * 2 ** intento))
        raise AssertionError

    def liberar(self, ruta: Path) -> None:
        if self.fuente == "hf" and self.borrar and str(ruta).startswith(str(self.hf_cache)):
            ruta.unlink(missing_ok=True)


def leer_filas(ruta: Path, filas: np.ndarray, ids: np.ndarray):
    """Genera (bytes, id) de las filas pedidas de un parquet, leyendo solo los row groups necesarios."""
    with pq.ParquetFile(ruta) as pf:
        limites = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
        orden = np.argsort(filas, kind="stable")
        filas, ids = filas[orden], ids[orden]
        grupo = np.searchsorted(limites, filas, side="right") - 1
        for g in np.unique(grupo):
            columna = pf.read_row_group(int(g), columns=["image"]).column(0)
            for fila, i in zip(filas[grupo == g], ids[grupo == g]):
                yield columna[int(fila - limites[g])].as_py()["bytes"], int(i)


class FlujoMuestras(IterableDataset):
    def __init__(self, tareas: list[tuple[str, np.ndarray, np.ndarray]], lector: Lector, transform,
                 buffer: int = 0, semilla: int = 0, tamano: int = 224, corrupta_cada: int | None = None):
        self.tareas, self.lector, self.transform = tareas, lector, transform
        self.buffer, self.semilla, self.tamano, self.corrupta_cada = buffer, semilla, tamano, corrupta_cada

    def _procesar(self, datos: bytes, i: int):
        try:
            if self.corrupta_cada and i % self.corrupta_cada == 0:
                datos = datos[: len(datos) // 3]  # Prueba: JPEG truncado.
            with Image.open(io.BytesIO(datos)) as im:
                return self.transform(im.convert("RGB")), i, True
        except Exception:  # noqa: BLE001 — la imagen se descarta y se cuenta en el loop principal.
            return torch.zeros(3, self.tamano, self.tamano), i, False

    def _descargas(self, mias):
        # Con fuente HF, baja el próximo archivo mientras se procesa el actual.
        if self.lector.fuente != "hf":
            for archivo, filas, ids in mias:
                yield self.lector.abrir(archivo), filas, ids
            return
        with ThreadPoolExecutor(1) as hilo:
            futuro = hilo.submit(self.lector.abrir, mias[0][0]) if mias else None
            for k, (_, filas, ids) in enumerate(mias):
                ruta = futuro.result()
                if k + 1 < len(mias):
                    futuro = hilo.submit(self.lector.abrir, mias[k + 1][0])
                yield ruta, filas, ids

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        torch.manual_seed(self.semilla * 1000 + wid)  # Crops y jitter reproducibles por worker.
        rng = np.random.default_rng([self.semilla, wid])
        mias = self.tareas[wid::nw]
        capacidad = max(1, self.buffer // nw) if self.buffer else 0
        pileta: list[tuple[bytes, int]] = []
        for ruta, filas, ids in self._descargas(mias):
            for datos, i in leer_filas(ruta, filas, ids):
                if not capacidad:
                    yield self._procesar(datos, i)
                    continue
                pileta.append((datos, i))
                if len(pileta) >= capacidad:
                    j = int(rng.integers(len(pileta)))
                    pileta[j], pileta[-1] = pileta[-1], pileta[j]
                    yield self._procesar(*pileta.pop())
            self.lector.liberar(ruta)
        rng.shuffle(pileta)
        for datos, i in pileta:
            yield self._procesar(datos, i)


def tareas_por_archivo(archivos: np.ndarray, filas: np.ndarray, ids: np.ndarray,
                       orden_archivos: list[str] | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Agrupa las muestras por archivo. `orden_archivos` fija el orden de recorrido (si no, alfabético)."""
    grupos = {}
    for a in np.unique(archivos):
        m = archivos == a
        grupos[str(a)] = (filas[m], ids[m])
    orden = orden_archivos if orden_archivos is not None else sorted(grupos)
    return [(a, *grupos[a]) for a in orden if a in grupos]


def ram_usada_gb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:  # noqa: BLE001
        return None
