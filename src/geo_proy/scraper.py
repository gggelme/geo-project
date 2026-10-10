
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

REPO_ID = "josefbednar/world-streetview-500k"
RUTA_DATASET = Path(r"E:\google_street_view_git")
NUM_WORKERS = 4

RUTA_DATASET.mkdir(parents=True, exist_ok=True)

api = HfApi()

archivos = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset",
)

parquets = [
    archivo
    for archivo in archivos
    if archivo.startswith("data/") and archivo.endswith(".parquet")
]

parquets_pendientes = []

for parquet in parquets:
    nombre = Path(parquet).name
    ruta_local = RUTA_DATASET / "data" / nombre

    if not ruta_local.exists():
        parquets_pendientes.append(parquet)

print(f"Parquets encontrados: {len(parquets)}")
print(f"Ya descargados: {len(parquets) - len(parquets_pendientes)}")
print(f"Pendientes: {len(parquets_pendientes)}")


def descargar(parquet):
    nombre = Path(parquet).name
    hilo = threading.current_thread().name

    print(f"[INICIO] {nombre} | {hilo}")

    hf_hub_download(
        repo_id=REPO_ID,
        filename=parquet,
        repo_type="dataset",
        local_dir=RUTA_DATASET,
    )

    print(f"[FIN] {nombre} | {hilo}")

    return f"[DESCARGADO] {nombre}"


with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futuros = [
        executor.submit(descargar, parquet)
        for parquet in parquets_pendientes
    ]

    for i, futuro in enumerate(as_completed(futuros), start=1):
        try:
            resultado = futuro.result()
            print(f"{i}/{len(parquets_pendientes)} {resultado}")

        except Exception as e:
            print(f"[ERROR] {e}")


print("\nDescarga terminada.")
print(f"Archivos guardados en: {RUTA_DATASET}")
