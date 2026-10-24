"""Rutas del proyecto y descubrimiento de corridas y datasets."""
from pathlib import Path
import json


def raiz_proyecto():
    raiz = next((p for p in Path(__file__).resolve().parents
                 if (p / "pyproject.toml").is_file()), None)
    if raiz is None:
        raise FileNotFoundError("No se encontró pyproject.toml subiendo desde el módulo.")
    return raiz


def _leer_json(ruta):
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def listar_corridas():
    """Corridas en outputs/<exp>/<timestamp>/ con checkpoint y config. Más recientes primero."""
    outputs = raiz_proyecto() / "outputs"
    corridas = []
    for config in outputs.glob("*/*/config_entrenamiento.json"):
        carpeta = config.parent
        checkpoints = sorted(carpeta.glob("mejor_*.pt")) or sorted(carpeta.glob("*.pt"))
        if not checkpoints:
            continue
        metricas = _leer_json(carpeta / "metricas_validation.json") or {}
        cfg = _leer_json(config) or {}
        partes = [f"{carpeta.parent.name}/{carpeta.name}"]
        if "accuracy_pais" in metricas:
            partes.append(f"acc país {metricas['accuracy_pais']:.1%}")
        if "filas_train" in cfg:
            partes.append(f"train {cfg['filas_train']:,}")
        corridas.append({"etiqueta": " · ".join(partes), "carpeta": carpeta,
                         "checkpoint": checkpoints[0]})
    corridas.sort(key=lambda c: c["carpeta"].name, reverse=True)
    return corridas


def listar_datasets():
    base = raiz_proyecto() / "datasets"
    rutas = sorted(p for p in base.rglob("*.parquet") if not p.name.endswith("_ids.parquet"))
    return [{"etiqueta": str(p.relative_to(base)), "ruta": p} for p in rutas]
