"""Réplicas con semillas y salida tabular (CSV)."""

import csv
import dataclasses
from pathlib import Path

import numpy as np

from .config import Config
from .motor import Simulacion


def _etiqueta(indice: int, n: int) -> str:
    """Etiqueta 1-based de un índice de columna; el índice n es el vacío."""
    return "vac" if indice == n else str(indice + 1)


def aplanar(resultado: dict) -> dict:
    """Aplana un resultado en una fila: `C_occ[0, 1]` -> columna `C_occ_1_2`.

    Los índices empiezan en 1 y la columna del vacío se llama `vac`
    (`C_full_1_vac`, `H_2_vac`). La serie temporal no se incluye.
    """
    n = len(resultado["poblaciones"])
    fila = {}
    for clave, valor in resultado.items():
        if clave == "serie":
            continue
        arr = np.asarray(valor) if isinstance(valor, (list, tuple, np.ndarray)) else None
        if arr is None or arr.ndim == 0:
            fila[clave] = valor.item() if isinstance(valor, np.generic) else valor
        elif arr.ndim == 1:
            for i, x in enumerate(arr):
                fila[f"{clave}_{_etiqueta(i, n)}"] = x.item()
        else:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    fila[f"{clave}_{_etiqueta(i, n)}_{_etiqueta(j, n)}"] = arr[i, j].item()
    return fila


def ejecutar_replicas(cfg: Config, semillas, salida=None) -> list:
    """Ejecuta una réplica por semilla y devuelve una lista de filas (dicts planos).

    Si se da `salida`, escribe además un CSV con una fila por réplica.
    La lista se puede pasar directamente a `pandas.DataFrame`.
    """
    filas = []
    for semilla in semillas:
        cfg_r = dataclasses.replace(cfg, seed=int(semilla))
        filas.append(aplanar(Simulacion(cfg_r).ejecutar()))
    if salida is not None:
        escribir_csv(filas, salida)
    return filas


def escribir_csv(filas: list, ruta) -> None:
    """Escribe las filas en un CSV (crea la carpeta si no existe)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = list(dict.fromkeys(c for f in filas for c in f))
    with open(ruta, "w", newline="", encoding="utf-8") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
