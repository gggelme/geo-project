"""Exporta un modelo de una corrida de src/entrenamiento al formato que lista el explorer.

Uso: .venv/Scripts/python.exe scripts/exportar_a_explorer.py --corrida outputs/entrenamiento_2m/prueba_200k
         [--modelo mejor | snapshot_corto_4 | ...]
Crea outputs/<carpeta padre de la corrida>_explorer/<corrida>_<modelo>/ con config_entrenamiento.json,
mejor_resnet18_celdas.pt (config + model_state_dict), prototipos.csv y metricas_validation.json.
La corrida original no se modifica.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entrenamiento.datos import transformaciones  # noqa: E402
from src.entrenamiento.entrenamiento import resolver  # noqa: E402


def plantilla_explorer(raiz: Path) -> dict:
    """Config de la corrida por cluster más reciente del explorer (para pais_a_continente y ruta_train)."""
    candidatas = []
    for ruta in (raiz / "outputs").glob("*/*/config_entrenamiento.json"):
        cfg = json.loads(ruta.read_text(encoding="utf-8"))
        if cfg.get("tarea") == "clasificacion_celdas_voronoi":
            candidatas.append((ruta.stat().st_mtime, cfg))
    if not candidatas:
        raise FileNotFoundError("No hay corridas por cluster del explorer para tomar pais_a_continente.")
    return max(candidatas, key=lambda c: c[0])[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrida", required=True)
    ap.add_argument("--modelo", default="mejor", help="mejor o el nombre de un snapshot (sin .pt)")
    args = ap.parse_args()
    corrida = resolver(args.corrida)
    raiz = resolver(".")
    cfg = json.loads((corrida / "config.json").read_text(encoding="utf-8"))
    # Corridas de Colab: la validación quedó con la ruta de la máquina (/content/...); se usa la local por defecto.
    if cfg.get("val_global") and not resolver(cfg["val_global"]).exists():
        from src.entrenamiento.config import Config
        cfg["val_global"] = Config.__dataclass_fields__["val_global"].default
    ruta_pt = corrida / "mejor.pt" if args.modelo == "mejor" else corrida / "snapshots" / f"{args.modelo}.pt"
    estado = torch.load(ruta_pt, map_location="cpu", weights_only=False)["modelo"]
    num_clases = int(estado["fc.1.weight"].shape[0])

    ruta_proto = resolver(cfg["prototipos"])
    plantilla = plantilla_explorer(raiz)
    tr, val = transformaciones(cfg["tamano_imagen"])
    destino = raiz / "outputs" / f"{corrida.parent.name}_explorer" / f"{corrida.name}_{args.modelo}"
    destino.mkdir(parents=True, exist_ok=True)

    finales = corrida / "metricas_finales.csv"
    n_train = None
    log = corrida / "log.jsonl"
    if log.exists():
        for linea in log.read_text(encoding="utf-8").splitlines():
            registro = json.loads(linea)
            if registro.get("tipo") == "inicio_sesion":
                n_train = registro.get("train_imagenes")
    config_explorer = {
        "arquitectura": "resnet18", "tarea": "clasificacion_celdas_voronoi", "weights": "IMAGENET1K_V1",
        "modo": "finetuning", "num_clases": num_clases, "columna_etiqueta": "ind_cluster",
        "prototipos": {"ruta": str(ruta_proto), "sha256": hashlib.sha256(ruta_proto.read_bytes()).hexdigest()},
        "etiqueta": "calculada al vuelo: prototipo más cercano por distancia angular",
        "ruta_train": plantilla.get("ruta_train"),  # Solo para nombrar los clusters por su composición.
        "ruta_validation": str(resolver(cfg["val_global"])) if cfg.get("val_global") else None,
        "pais_a_continente": plantilla["pais_a_continente"],
        "tamano_imagen": cfg["tamano_imagen"], "dropout": cfg["dropout"], "tau_km": cfg["tau_km"],
        "loss": f"focal haversine (gamma={cfg['gamma_focal']}, tau={cfg['tau_km']} km) + SAM (rho={cfg['rho_sam']})",
        "amp": False, "filas_train": n_train,
        "preprocesamiento_train": str(tr), "preprocesamiento_val": str(val),
        "origen": {"corrida": str(corrida), "modelo": args.modelo, "huella": cfg.get("huella")},
    }
    (destino / "config_entrenamiento.json").write_text(json.dumps(config_explorer, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    torch.save({"config": config_explorer, "model_state_dict": estado}, destino / "mejor_resnet18_celdas.pt")
    shutil.copy(ruta_proto, destino / "prototipos.csv")

    metricas = {}
    if finales.exists():
        df = pd.read_csv(finales)
        fila = df[(df.modelo == ruta_pt.stem) & (df.tabla == "val_global")]
        if len(fila):
            f = fila.iloc[0]
            metricas = {"n_validation": None, "accuracy_top1": f.top1, "accuracy_top5": f.top5,
                        "error_mediano_km": f.km_mediana, "error_medio_km": f.km_media,
                        "error_mediano_km_mbr": f.km_mediana_mbr, "loss_haversine_validation": f.ce_haversine}
    (destino / "metricas_validation.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    print(f"Exportado a {destino}")


if __name__ == "__main__":
    main()
