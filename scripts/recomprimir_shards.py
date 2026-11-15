"""Recomprime los parquets de train y val a imágenes más chicas para que entren en el disco de Colab.

El modelo ve 224 px: el RandomResizedCrop (60-100% del área) y el Resize de validación siempre achican, así que
guardar a 384 px no le quita nada a lo que llega a la red (el recorte más chico, ~297 px, se sigue achicando a
224). De ~181 GB pasa a ~73 GB (384 px, JPEG calidad 90).

Se conserva el orden de las filas y todas las columnas: splits.parquet, el manifest y el sampler siguen valiendo.
Cada archivo se escribe de forma atómica con metadatos (lado, calidad, filas); los que ya están hechos se saltean,
así que se puede cortar y relanzar. Con fuente HF, cada original se baja, se recomprime y se borra.

Uso (Colab):
    python scripts/recomprimir_shards.py --destino /content/shards/data --tmp /content/shards_tmp \
        --splits /content/insumos/splits.parquet --revision <commit del dataset> --procesos 11
Uso (prueba local, desde la copia en disco):
    python scripts/recomprimir_shards.py --fuente local --origen ../world-streetview-500k/data --destino <carpeta> \
        --archivos train-00000-of-00393.parquet
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

VERSION = "1"
ARGS = None  # Se completa en cada proceso (initializer).


def metadatos_esperados(filas: int) -> dict[bytes, bytes]:
    return {b"recomprimido_version": VERSION.encode(), b"recomprimido_lado": str(ARGS.lado).encode(),
            b"recomprimido_calidad": str(ARGS.calidad).encode(), b"recomprimido_filas": str(filas).encode()}


def ya_hecho(destino: Path) -> bool:
    if not destino.exists():
        return False
    try:
        with pq.ParquetFile(destino) as pf:
            meta = pf.schema_arrow.metadata or {}
            filas = pf.metadata.num_rows
    except Exception:  # noqa: BLE001 — archivo a medio escribir o dañado: se rehace.
        return False
    esperado = metadatos_esperados(filas)
    return all(meta.get(k) == v for k, v in esperado.items())


def recomprimir_imagen(datos: bytes) -> tuple[bytes, bool]:
    """(bytes nuevos, ok). Si no decodifica o ya es chica, devuelve los originales."""
    try:
        with Image.open(io.BytesIO(datos)) as im:
            im = im.convert("RGB")
            ancho, alto = im.size
            escala = ARGS.lado / min(ancho, alto)
            if escala >= 1:
                return datos, True
            im = im.resize((round(ancho * escala), round(alto * escala)), Image.BICUBIC)
            salida = io.BytesIO()
            im.save(salida, "JPEG", quality=ARGS.calidad)
            return salida.getvalue(), True
    except Exception:  # noqa: BLE001 — el entrenamiento ya cuenta y descarta las que no decodifican.
        return datos, False


def obtener_original(archivo: str) -> Path:
    if ARGS.fuente == "local":
        return Path(ARGS.origen) / archivo
    from huggingface_hub import hf_hub_download

    for intento in range(8):
        try:
            return Path(hf_hub_download(ARGS.hf_repo, f"data/{archivo}", repo_type="dataset",
                                        revision=ARGS.revision, local_dir=ARGS.tmp))
        except Exception as e:  # noqa: BLE001 — red: reintento con espera creciente.
            if intento == 7:
                raise RuntimeError(f"No se pudo bajar {archivo}") from e
            time.sleep(min(300, 10 * 2 ** intento))
    raise AssertionError


def procesar(archivo: str) -> tuple[str, str, int, int, float, float]:
    destino = Path(ARGS.destino) / archivo
    if ya_hecho(destino):
        return archivo, "ya estaba", 0, 0, 0.0, 0.0
    t0 = time.time()
    original = obtener_original(archivo)
    t_bajada = time.time() - t0
    tmp = destino.with_suffix(".parquet.tmp")
    fallidas = 0
    with pq.ParquetFile(original) as pf:
        schema = pf.schema_arrow
        filas = pf.metadata.num_rows
        schema = schema.with_metadata({**(schema.metadata or {}), **metadatos_esperados(filas)})
        indice = schema.get_field_index("image")
        with pq.ParquetWriter(tmp, schema, compression="none") as writer:
            for g in range(pf.num_row_groups):
                tabla = pf.read_row_group(g)
                imagenes = tabla.column("image").to_pylist()
                nuevas = []
                for im in imagenes:
                    if im is None or im.get("bytes") is None:
                        nuevas.append(im)
                        continue
                    datos, ok = recomprimir_imagen(im["bytes"])
                    fallidas += not ok
                    nuevas.append({**im, "bytes": datos})
                columna = pa.array(nuevas, type=schema.field("image").type)
                tabla = tabla.set_column(indice, schema.field("image"), columna)
                writer.write_table(tabla.replace_schema_metadata(schema.metadata))
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, destino)
    if ARGS.fuente == "hf":
        original.unlink(missing_ok=True)
    return archivo, "ok", filas, fallidas, t_bajada, time.time() - t0 - t_bajada


def iniciar(args):
    global ARGS
    ARGS = args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--destino", required=True)
    ap.add_argument("--fuente", choices=["hf", "local"], default="hf")
    ap.add_argument("--origen", help="Carpeta con los parquets originales (fuente local)")
    ap.add_argument("--tmp", default="/content/shards_tmp", help="Dónde bajar los originales de HF")
    ap.add_argument("--hf-repo", default="josefbednar/world-streetview-500k")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--splits", help="splits.parquet: se recomprimen los archivos de train y val")
    ap.add_argument("--archivos", nargs="*", help="Lista explícita de archivos (en vez de --splits)")
    ap.add_argument("--lado", type=int, default=384, help="Lado menor de salida (px)")
    ap.add_argument("--calidad", type=int, default=90)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    if args.archivos:
        archivos = args.archivos
    elif args.splits:
        splits = pq.read_table(args.splits, columns=["split", "archivo"]).to_pandas()
        archivos = sorted(splits.loc[splits.split.isin(["train", "val"]), "archivo"].unique())
    else:
        sys.exit("Indicar --splits o --archivos.")
    Path(args.destino).mkdir(parents=True, exist_ok=True)
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    print(f"{len(archivos)} archivos -> {args.destino} ({args.lado} px, calidad {args.calidad}, "
          f"{args.procesos} procesos)", flush=True)
    t0, hechos, filas_total, fallidas_total = time.time(), 0, 0, 0
    with Pool(args.procesos, initializer=iniciar, initargs=(args,)) as pool:
        for archivo, estado, filas, fallidas, t_bajada, t_proc in pool.imap_unordered(procesar, archivos):
            hechos += 1
            filas_total += filas
            fallidas_total += fallidas
            transcurrido = (time.time() - t0) / 60
            print(f"[{hechos}/{len(archivos)}] {archivo}: {estado}"
                  + (f" ({filas} filas, bajada {t_bajada:.0f}s, recompresión {t_proc:.0f}s"
                     + (f", {fallidas} no decodificaron" if fallidas else "") + ")" if estado == "ok" else "")
                  + f" | {transcurrido:.1f} min", flush=True)
    iniciar(args)
    faltan = [a for a in archivos if not ya_hecho(Path(args.destino) / a)]
    if faltan:
        sys.exit(f"Quedaron {len(faltan)} archivos sin recomprimir: {faltan[:5]}")
    gb = sum((Path(args.destino) / a).stat().st_size for a in archivos) / 1e9
    print(f"Listo: {len(archivos)} archivos, {gb:.1f} GB, {(time.time() - t0) / 60:.1f} min "
          f"({fallidas_total} imágenes no decodificaron y quedaron como estaban)", flush=True)


if __name__ == "__main__":
    main()
