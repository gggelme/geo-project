"""Entrena o retoma una corrida (src/entrenamiento). Relanzar el mismo comando sigue desde el último checkpoint.

Uso (desde la raíz del proyecto):
    .venv/Scripts/python.exe scripts/entrenar.py --config configs/prueba_200k.json
    .venv/Scripts/python.exe scripts/entrenar.py --config configs/humo.json --cambiar max_horas=0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entrenamiento.config import Config  # noqa: E402
from src.entrenamiento.entrenamiento import entrenar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cambiar", nargs="*", default=[], metavar="CAMPO=VALOR",
                    help="Sobrescribe campos de la configuración (valor en JSON; texto sin comillas si no parsea)")
    args = ap.parse_args()
    cambios = {}
    for item in args.cambiar:
        campo, _, valor = item.partition("=")
        try:
            cambios[campo] = json.loads(valor)
        except json.JSONDecodeError:
            cambios[campo] = valor
    resultado = entrenar(Config.desde_json(args.config, **cambios))
    print(f"Resultado: {resultado}")
    sys.exit(0 if resultado == "terminado" else 3)  # 3 = pausado (se retoma relanzando).


if __name__ == "__main__":
    main()
