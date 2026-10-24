"""Lectura de metadatos e imágenes desde parquet sin cargar todo en memoria."""
from io import BytesIO
from functools import lru_cache
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .paths import raiz_proyecto


def cargar_metadatos(ruta):
    pf = pq.ParquetFile(Path(ruta))
    columnas = [c for c in pf.schema_arrow.names if c != "image"]
    meta = pf.read(columns=columnas).to_pandas()
    meta["country_code"] = meta["country_code"].str.strip().str.upper()
    meta.insert(0, "idx_fila", np.arange(len(meta)))
    return meta


@lru_cache(maxsize=4)
def _archivo(ruta):
    pf = pq.ParquetFile(ruta)
    offsets = np.concatenate([[0], np.cumsum([pf.metadata.row_group(i).num_rows
                                              for i in range(pf.num_row_groups)])])
    return pf, offsets


def leer_imagen(ruta, idx):
    """Imagen RGB de la fila `idx` del parquet."""
    pf, offsets = _archivo(str(ruta))
    grupo = int(np.searchsorted(offsets, idx, side="right") - 1)
    tabla = pf.read_row_group(grupo, columns=["image"])
    contenido = tabla.column("image")[int(idx - offsets[grupo])].as_py()
    if contenido.get("bytes") is not None:
        fuente = BytesIO(contenido["bytes"])
    else:
        p = Path(contenido["path"])
        fuente = p if p.is_absolute() else raiz_proyecto() / p
    with Image.open(fuente) as im:
        return im.convert("RGB")
