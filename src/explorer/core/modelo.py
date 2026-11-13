"""Carga de checkpoints y registro de arquitecturas."""
import re
import torch
from torch import nn
from torchvision import models, transforms

# Arquitectura -> (constructor, capa convolucional final para Grad-CAM, capa de features)
ARQUITECTURAS = {
    "resnet18": (lambda n: models.resnet18(weights=None, num_classes=n), "layer4", "avgpool"),
}
MEDIA_IMAGENET, STD_IMAGENET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def tarea(config):
    """'cluster' para los clasificadores de celdas (prototipos), 'pais' para los de país."""
    if config.get("tarea") == "clasificacion_celdas_voronoi" or "prototipos" in config:
        return "cluster"
    return "pais"


def num_clases(config):
    return int(config["num_clases"]) if tarea(config) == "cluster" else len(config["paises"])


def normalizacion(config):
    """(media, desvío) del preprocesamiento de validación con el que se entrenó."""
    texto = config.get("preprocesamiento_val", "")
    m = re.search(r"Normalize\(mean=[\[(]([^\])]*)[\])],\s*std=[\[(]([^\])]*)[\])]", texto)
    if m:
        return tuple(tuple(float(v) for v in g.split(",")) for g in m.groups())
    if str(config.get("weights", "")).upper().startswith("IMAGENET"):
        return MEDIA_IMAGENET, STD_IMAGENET
    return (0.5,) * 3, (0.5,) * 3


def _construir(config, estado):
    n = num_clases(config)
    modelo = ARQUITECTURAS[config["arquitectura"]][0](n)
    if "fc.1.weight" in estado:  # cabeza Dropout + Linear de los clasificadores de celdas
        modelo.fc = nn.Sequential(nn.Dropout(float(config.get("dropout", 0.0))),
                                  nn.Linear(modelo.fc.in_features, n))
    return modelo


def cargar_modelo(ruta_checkpoint, device="cpu"):
    """Devuelve (modelo en eval, config, transform de validación)."""
    estado = torch.load(ruta_checkpoint, map_location="cpu", weights_only=True)
    config = estado["config"]
    if config["arquitectura"] not in ARQUITECTURAS:
        raise ValueError(f"Arquitectura no registrada: {config['arquitectura']}")
    modelo = _construir(config, estado["model_state_dict"])
    modelo.load_state_dict(estado["model_state_dict"])
    modelo.to(device).eval()
    tam = config["tamano_imagen"]
    media, std = normalizacion(config)
    transform = transforms.Compose([
        transforms.Resize(tam),
        transforms.CenterCrop(tam),
        transforms.ToTensor(),
        transforms.Normalize(media, std),
    ])
    return modelo, config, transform


def capa_gradcam(modelo, config):
    return dict(modelo.named_modules())[ARQUITECTURAS[config["arquitectura"]][1]]


def capa_features(modelo, config):
    """Módulo cuya salida es el vector de features previo al clasificador."""
    return dict(modelo.named_modules())[ARQUITECTURAS[config["arquitectura"]][2]]
