"""Grad-CAM con hooks sobre la última capa convolucional."""
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from .modelo import capa_gradcam


def gradcam(modelo, config, tensor, clase, device="cpu"):
    """Mapa (H,W) en [0,1] para `clase` (índice) sobre un tensor (3,H,W). fp32, sin autocast."""
    capa = capa_gradcam(modelo, config)
    guardado = {}
    h1 = capa.register_forward_hook(lambda m, i, o: guardado.__setitem__("act", o))
    h2 = capa.register_full_backward_hook(lambda m, gi, go: guardado.__setitem__("grad", go[0]))
    try:
        modelo.zero_grad(set_to_none=True)
        x = tensor.unsqueeze(0).to(device).requires_grad_(True)
        with torch.enable_grad():
            logits = modelo(x)
            logits[0, int(clase)].backward()
        pesos = guardado["grad"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((pesos * guardado["act"]).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=tuple(tensor.shape[1:]), mode="bilinear",
                                              align_corners=False)[0, 0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    rango = cam.max() - cam.min()
    return (cam - cam.min()) / rango if rango > 0 else np.zeros_like(cam)


def superponer(imagen, mapa, alpha=0.5):
    """Superpone el mapa (H,W) sobre una PIL image, reescalada al tamaño del mapa."""
    imagen = imagen.resize(mapa.shape[::-1])
    calor = (colormaps["jet"](mapa)[..., :3] * 255).astype(np.float32)
    mezcla = (1 - alpha) * np.asarray(imagen, dtype=np.float32) + alpha * calor
    return Image.fromarray(mezcla.astype(np.uint8))
