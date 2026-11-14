"""Planificador de rondas balanceadas por celda.

Una ronda es un tramo de `tamano_ronda` muestras. No es una época (una época recorre todos los datos; una
ronda no): es la unidad de evaluación, plateau y early stopping. Cada ronda está balanceada por sí misma:

- Cuota por celda por water-filling: todas las celdas aportan lo mismo, salvo las que no tienen imágenes
  suficientes; su sobrante se reparte entre las demás.
- Límite de repetición POR RONDA: cada imagen sale a lo sumo una vez por ronda (un panorama, hasta tantas veces
  como vistas usables tenga, normalmente 4). Así una celda chica aporta todas sus imágenes en cada ronda y una
  grande aporta un subconjunto distinto cada vez. El límite es fijo, de modo que la mezcla de celdas no cambia a
  lo largo del entrenamiento (un tope global de pasadas haría desaparecer a las celdas chicas de las últimas
  rondas; existe como opción, `r_max`, desactivada por defecto).
- Cada celda recorre una permutación de sus panoramas, y al agotarla otra nueva: un panorama no se repite
  hasta que pasaron todos los de su celda.
- En cada aparición de un panorama se toma la vista siguiente (desde una vista inicial al azar), así que sus
  primeras apariciones usan vistas distintas.

Es determinístico: la ronda r depende solo de la semilla y de las rondas anteriores. Para retomar se
reconstruye el estado repitiendo la planificación de las rondas ya hechas (`avanzar_hasta`).
"""
from __future__ import annotations

import hashlib

import numpy as np


def water_filling(capacidad: np.ndarray, total: int) -> np.ndarray:
    """Reparte `total` en partes lo más iguales posible sin superar la capacidad de cada celda."""
    capacidad = np.asarray(capacidad, dtype=np.int64)
    if capacidad.sum() <= total:
        return capacidad.copy()
    cuota = np.zeros_like(capacidad)
    restante, activas = total, capacidad > 0
    while restante > 0 and activas.any():
        parte = restante // activas.sum()
        if parte == 0:  # Quedan menos unidades que celdas: una a cada una de las de más capacidad libre.
            libres = np.flatnonzero(activas)
            orden = libres[np.argsort(-(capacidad - cuota)[libres], kind="stable")][:restante]
            cuota[orden] += 1
            break
        suma = np.minimum(capacidad - cuota, parte) * activas
        cuota += suma
        restante -= int(suma.sum())
        activas = (capacidad - cuota) > 0
    return cuota


class Planificador:
    def __init__(self, celda_pano: np.ndarray, inicio_filas: np.ndarray, n_filas: np.ndarray, n_celdas: int,
                 tamano_ronda: int, semilla: int, r_max: int | None = None):
        self.celda_pano = np.asarray(celda_pano, dtype=np.int64)
        self.inicio_filas = np.asarray(inicio_filas, dtype=np.int64)
        self.n_filas = np.asarray(n_filas, dtype=np.int64)  # Vistas usables de cada panorama.
        self.n_celdas, self.tamano_ronda, self.semilla = n_celdas, tamano_ronda, semilla
        self.r_max = r_max if r_max is not None else np.iinfo(np.int32).max  # Sin tope global.
        n_panos = len(self.celda_pano)
        self.vista_inicial = np.random.default_rng([semilla, 0]).integers(0, 4, n_panos)
        self.usos = np.zeros(n_panos, dtype=np.int32)
        self.panos_celda = [np.flatnonzero(self.celda_pano == c) for c in range(n_celdas)]
        self.vuelta = np.zeros(n_celdas, dtype=np.int64)
        self.posicion = np.zeros(n_celdas, dtype=np.int64)
        self.permutacion = [self._permutar(c, 0) for c in range(n_celdas)]
        self.rondas = 0  # Rondas ya planificadas.

    def _permutar(self, celda: int, vuelta: int) -> np.ndarray:
        return np.random.default_rng([self.semilla, 1, celda, vuelta]).permutation(self.panos_celda[celda])

    def _limite_ronda(self) -> np.ndarray:
        # Apariciones permitidas de cada panorama en esta ronda: una por vista, sin pasar el tope global.
        return np.clip(np.minimum(self.n_filas, self.r_max - self.usos), 0, None)

    def _tomar(self, celda: int, cantidad: int, limite: np.ndarray) -> list[int]:
        elegidos, en_ronda = [], {}
        while len(elegidos) < cantidad:
            perm = self.permutacion[celda]
            if self.posicion[celda] >= len(perm):
                self.vuelta[celda] += 1
                self.permutacion[celda] = self._permutar(celda, int(self.vuelta[celda]))
                self.posicion[celda] = 0
                continue
            p = int(perm[self.posicion[celda]])
            self.posicion[celda] += 1
            if en_ronda.get(p, 0) < limite[p]:
                elegidos.append(p)
                en_ronda[p] = en_ronda.get(p, 0) + 1
        return elegidos

    def siguiente_ronda(self) -> np.ndarray:
        """Índices de fila (en la tabla de train) de la próxima ronda. Vacío si ya no quedan datos."""
        limite = self._limite_ronda()
        capacidad = np.bincount(self.celda_pano, weights=limite, minlength=self.n_celdas).astype(np.int64)
        cuotas = water_filling(capacidad, self.tamano_ronda)
        filas = []
        for celda in np.flatnonzero(cuotas):
            for p in self._tomar(int(celda), int(cuotas[celda]), limite):
                vista = (self.vista_inicial[p] + self.usos[p]) % self.n_filas[p]  # Vista siguiente.
                filas.append(self.inicio_filas[p] + vista)
                self.usos[p] += 1
        self.rondas += 1
        return np.asarray(filas, dtype=np.int64)

    def avanzar_hasta(self, rondas: int) -> None:
        while self.rondas < rondas:
            self.siguiente_ronda()

    def huella(self) -> str:
        """Resumen del estado, para verificar que la reconstrucción al retomar coincide."""
        return hashlib.sha256(self.usos.tobytes() + self.posicion.tobytes() + self.vuelta.tobytes()).hexdigest()[:16]

    def resumen(self) -> dict:
        return {"rondas": self.rondas,
                "usos_medios_por_celda": [round(float(self.usos[p].mean()), 2) if len(p) else 0.0
                                          for p in self.panos_celda]}
