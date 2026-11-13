"""Rutas del proyecto y descubrimiento de corridas y datasets."""
from pathlib import Path, PureWindowsPath
import json


def raiz_proyecto():
    raiz = next((p for p in Path(__file__).resolve().parents
                 if (p / "pyproject.toml").is_file()), None)
    if raiz is None:
        raise FileNotFoundError("No se encontró pyproject.toml subiendo desde el módulo.")
    return raiz


def resolver_ruta(ruta):
    """Ruta de un config (absoluta, quizás de otra máquina) dentro de este proyecto, o None."""
    if not ruta:
        return None
    p = Path(ruta)
    if p.is_file():
        return p
    raiz = raiz_proyecto()
    if not p.is_absolute() and (raiz / p).is_file():
        return raiz / p
    partes = PureWindowsPath(ruta).parts if "\\" in str(ruta) else p.parts
    for ancla in ("datasets", "src", "outputs"):  # reubica a partir de la primera carpeta conocida
        if ancla in partes:
            candidata = raiz.joinpath(*partes[partes.index(ancla):])
            if candidata.is_file():
                return candidata
    return None


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
        es_cluster = cfg.get("tarea") == "clasificacion_celdas_voronoi" or "prototipos" in cfg
        partes = [("[cluster] " if es_cluster else "[país] ") + f"{carpeta.parent.name}/{carpeta.name}"]
        if "accuracy_pais" in metricas:
            partes.append(f"acc país {metricas['accuracy_pais']:.1%}")
        if "accuracy_top1" in metricas:
            partes.append(f"acc {metricas['accuracy_top1']:.1%}")
        if "error_mediano_km" in metricas:
            partes.append(f"mediana {metricas['error_mediano_km']:,.0f} km")
        if "filas_train" in cfg:
            partes.append(f"train {cfg['filas_train']:,}")
        corridas.append({"etiqueta": " · ".join(partes), "carpeta": carpeta,
                         "checkpoint": checkpoints[0], "ruta_validation": cfg.get("ruta_validation")})
    corridas.sort(key=lambda c: c["carpeta"].name, reverse=True)
    return corridas


def listar_datasets():
    base = raiz_proyecto() / "datasets"
    rutas = sorted(p for p in base.rglob("*.parquet") if not p.name.endswith("_ids.parquet"))
    return [{"etiqueta": str(p.relative_to(base)), "ruta": p} for p in rutas]
