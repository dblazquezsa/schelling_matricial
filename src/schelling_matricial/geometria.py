"""Geometría de la retícula toroidal: vecindades, k y distancias."""

import numpy as np

VECINDADES = ("moore", "von_neumann")


def desplazamientos(vecindad: str, radio: int) -> np.ndarray:
    """Devuelve los desplazamientos (dx, dy) de la vecindad, sin el (0, 0).

    - Moore: max(|dx|, |dy|) <= radio, con k = (2r+1)^2 - 1.
    - von Neumann: |dx| + |dy| <= radio, con k = 2r(r+1).

    El resultado es un array entero de forma (k, 2).
    """
    if vecindad not in VECINDADES:
        raise ValueError(f"vecindad debe ser una de {VECINDADES}, no {vecindad!r}")
    if radio < 1:
        raise ValueError("el radio debe ser >= 1")
    lista = []
    for dx in range(-radio, radio + 1):
        for dy in range(-radio, radio + 1):
            if dx == 0 and dy == 0:
                continue
            if vecindad == "moore" or abs(dx) + abs(dy) <= radio:
                lista.append((dx, dy))
    return np.array(lista, dtype=np.int64)


def numero_vecinos(vecindad: str, radio: int) -> int:
    """k: número de vecinos de una casilla."""
    if vecindad == "moore":
        return (2 * radio + 1) ** 2 - 1
    if vecindad == "von_neumann":
        return 2 * radio * (radio + 1)
    raise ValueError(f"vecindad debe ser una de {VECINDADES}, no {vecindad!r}")


def comprobar_tamano(L: int, radio: int) -> None:
    """Exige L > 2r para que ningún vecino aparezca repetido en el toro."""
    if L <= 2 * radio:
        raise ValueError(f"se necesita L > 2r (L={L}, r={radio}): habría vecinos repetidos")


def diferencia_toroidal(a, b, L: int):
    """|a - b| en el toro de lado L, componente a componente (0 <= d <= L/2)."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % L
    return np.minimum(d, L - d)


def distancia_chebyshev(fila0, col0, filas, cols, L: int):
    """Distancia de Chebyshev toroidal entre (fila0, col0) y las casillas dadas."""
    return np.maximum(diferencia_toroidal(filas, fila0, L), diferencia_toroidal(cols, col0, L))


def son_vecinos(fila0, col0, filas, cols, L: int, vecindad: str, radio: int):
    """Máscara: qué casillas (filas, cols) son vecinas de (fila0, col0).

    Una casilla no es vecina de sí misma.
    """
    df = diferencia_toroidal(filas, fila0, L)
    dc = diferencia_toroidal(cols, col0, L)
    if vecindad == "moore":
        dentro = np.maximum(df, dc) <= radio
    else:
        dentro = (df + dc) <= radio
    return dentro & ((df + dc) > 0)


def contar_vecinos(reticula: np.ndarray, n: int, desp: np.ndarray) -> np.ndarray:
    """Campo de recuentos desde cero: cuentas[j] = número de vecinos de tipo j+1.

    Devuelve un array entero de forma (n, L, L). Se calcula sumando np.roll de
    la indicatriz de cada tipo sobre los desplazamientos de la vecindad.
    """
    L = reticula.shape[0]
    cuentas = np.zeros((n, L, L), dtype=np.int64)
    for j in range(n):
        indicatriz = (reticula == j + 1).astype(np.int64)
        for dx, dy in desp:
            # El vecino de x en x + d aporta a x: desplazamos el campo en -d.
            cuentas[j] += np.roll(indicatriz, shift=(-dx, -dy), axis=(0, 1))
    return cuentas
