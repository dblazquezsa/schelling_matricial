"""Utilidades comunes de las pruebas."""

import numpy as np

from schelling_matricial import Config, Simulacion


def trayectoria(cfg: Config, barridos: int = 30) -> list:
    """Retícula tras cada barrido (hasta `barridos` o hasta un punto fijo)."""
    sim = Simulacion(cfg)
    estados = [sim.reticula.copy()]
    for _ in range(barridos):
        movimientos = sim.barrido()
        estados.append(sim.reticula.copy())
        if movimientos == 0:
            break
    return estados


def iguales(t1: list, t2: list) -> bool:
    return len(t1) == len(t2) and all(np.array_equal(a, b) for a, b in zip(t1, t2))
