"""Extensión matricial del modelo de desplazamiento de Schelling (modelos P y S)."""

from .config import Config
from .experimentos import aplanar, ejecutar_replicas
from .motor import Simulacion, __version__
from .observables import calcular_observables, phi

__all__ = [
    "Config",
    "Simulacion",
    "ejecutar_replicas",
    "aplanar",
    "calcular_observables",
    "phi",
    "__version__",
]
