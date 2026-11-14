"""Entrenamiento del clasificador por celdas K=70 sobre el dataset completo (local o streaming desde HF).

Mismo código para la prueba local de 200k y para la corrida de 2M en Colab: solo cambia la configuración.
Punto de entrada: `entrenar(Config(...))` (ver entrenamiento.py) o `scripts/entrenar.py --config ...`.
"""
from .config import Config  # noqa: F401
