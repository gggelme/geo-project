"""Carga de checkpoints y registro de arquitecturas."""
import torch
from torchvision import models, transforms

# Arquitectura -> (constructor, capa convolucional final para Grad-CAM, capa de features)
ARQUITECTURAS = {
    "resnet18": (lambda n: models.resnet18(weights=None, num_classes=n), "layer4", "avgpool"),
}


def cargar_modelo(ruta_checkpoint, device="cpu"):
    """Devuelve (modelo en eval, config, transform de validación)."""
    estado = torch.load(ruta_checkpoint, map_location="cpu", weights_only=True)
    config = estado["config"]
    arquitectura = config["arquitectura"]
    if arquitectura not in ARQUITECTURAS:
        raise ValueError(f"Arquitectura no registrada: {arquitectura}")
    modelo = ARQUITECTURAS[arquitectura][0](len(config["paises"]))
    modelo.load_state_dict(estado["model_state_dict"])
    modelo.to(device).eval()
    tam = config["tamano_imagen"]
    transform = transforms.Compose([
        transforms.Resize(tam),
        transforms.CenterCrop(tam),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    return modelo, config, transform


def capa_gradcam(modelo, config):
    return dict(modelo.named_modules())[ARQUITECTURAS[config["arquitectura"]][1]]


def capa_features(modelo, config):
    """Módulo cuya salida es el vector de features previo al clasificador."""
    return dict(modelo.named_modules())[ARQUITECTURAS[config["arquitectura"]][2]]
