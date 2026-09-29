"""Configuración validada de una simulación y conversión a enteros exactos.

Aritmética exacta
-----------------
Las decisiones de los agentes (cómodo / incómodo, mejora) no se toman nunca
comparando números en coma flotante. `H`, `tau` e `iota` se convierten a
racionales con ``Fraction(str(x)).limit_denominator(10**6)``, se multiplican por
el mínimo común múltiplo `escala` de todos los denominadores y se guardan como
enteros ``int64``. Multiplicar todo por el mismo factor positivo no cambia la
dinámica.

Riesgo de desbordamiento: en el modelo P se comparan productos del tipo
``num_y * den_x``, con ``|num| <= k * max|H|`` y ``den <= k``, es decir,
valores de hasta ``k^2 * max|H_entero|``. Con ``k <= 48`` y coeficientes
moderados esto queda muy lejos de ``2^63``. Aun así se comprueba al crear la
configuración y se lanza ``OverflowError`` si se superara una cota prudente.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
from typing import Optional, Sequence, Union

import numpy as np

from .geometria import VECINDADES, comprobar_tamano, numero_vecinos

MODELOS = ("P", "S")
BUSQUEDAS = ("cercano", "aleatorio", "mejor")

_COTA_ENTEROS = 2**62

Numero = Union[int, float, Fraction]


def a_fraccion(x) -> Fraction:
    """Convierte un número a racional exacto (vía su representación decimal)."""
    return Fraction(str(x)).limit_denominator(10**6)


@dataclass
class Config:
    """Parámetros de una simulación.

    - `L`: lado de la retícula toroidal L×L.
    - `poblaciones`: (N_1, ..., N_n); el resto de casillas quedan vacías.
    - `H`: matriz n×(n+1) (la última columna es el vacío) o n×n (se añade una
      columna de ceros). `H[i, j]` es la incomodidad que un agente de tipo i+1
      asocia a un vecino de tipo j+1.
    - `tau`: umbral común (escalar) o por tipo (vector de longitud n).
    - `iota`: incomodidad de un agente aislado en el modelo P (escalar o vector).
    - `modelo`: "P" (promediado) o "S" (de suma).
    - `vecindad`: "moore" o "von_neumann"; `radio`: radio de la vecindad.
    - `busqueda`: regla de elección de destino ("cercano" en v0.1).
    - `T_max`: número máximo de barridos.
    - `seed`: semilla de `numpy.random.default_rng`.
    - `parar_en_repeticion`: parar si el estado al final de un barrido ya
      apareció antes.
    - `cada`: si no es None, se registran los observables cada `cada` barridos.
    """

    L: int
    poblaciones: Sequence[int]
    H: Sequence[Sequence[Numero]]
    tau: Union[Numero, Sequence[Numero]] = 0.5
    iota: Union[Numero, Sequence[Numero]] = 0
    modelo: str = "P"
    vecindad: str = "moore"
    radio: int = 1
    busqueda: str = "cercano"
    T_max: int = 1000
    seed: Optional[int] = None
    parar_en_repeticion: bool = True
    cada: Optional[int] = None

    # Derivados (se calculan en __post_init__)
    n: int = field(init=False, repr=False)
    k: int = field(init=False, repr=False)
    escala: int = field(init=False, repr=False)
    H_ent: np.ndarray = field(init=False, repr=False)
    tau_ent: np.ndarray = field(init=False, repr=False)
    iota_ent: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.poblaciones = tuple(int(N) for N in self.poblaciones)
        n = len(self.poblaciones)
        if n < 1:
            raise ValueError("hace falta al menos un tipo")
        if any(N < 0 for N in self.poblaciones):
            raise ValueError("las poblaciones deben ser >= 0")
        if self.L < 1:
            raise ValueError("L debe ser >= 1")
        if sum(self.poblaciones) > self.L * self.L:
            raise ValueError("la suma de poblaciones supera L^2")
        if self.modelo not in MODELOS:
            raise ValueError(f"modelo debe ser uno de {MODELOS}, no {self.modelo!r}")
        if self.vecindad not in VECINDADES:
            raise ValueError(f"vecindad debe ser una de {VECINDADES}, no {self.vecindad!r}")
        if self.busqueda not in BUSQUEDAS:
            raise ValueError(f"busqueda debe ser una de {BUSQUEDAS}, no {self.busqueda!r}")
        if self.T_max < 1:
            raise ValueError("T_max debe ser >= 1")
        if self.cada is not None and self.cada < 1:
            raise ValueError("cada debe ser None o >= 1")
        comprobar_tamano(self.L, self.radio)

        # Matriz H como racionales, con la columna de vacío.
        filas = [list(f) for f in self.H]
        if len(filas) != n:
            raise ValueError(f"H debe tener {n} filas (una por tipo), tiene {len(filas)}")
        for f in filas:
            if len(f) == n:
                f.append(0)
            elif len(f) != n + 1:
                raise ValueError(f"cada fila de H debe tener {n} o {n + 1} columnas")
        H_frac = [[a_fraccion(x) for x in f] for f in filas]
        tau_frac = _vector(self.tau, n, "tau")
        iota_frac = _vector(self.iota, n, "iota")

        todos = [x for f in H_frac for x in f] + tau_frac + iota_frac
        escala = lcm(*(x.denominator for x in todos))

        self.n = n
        self.k = numero_vecinos(self.vecindad, self.radio)
        self.escala = escala
        self.H_ent = np.array([[int(x * escala) for x in f] for f in H_frac], dtype=np.int64)
        self.tau_ent = np.array([int(x * escala) for x in tau_frac], dtype=np.int64)
        self.iota_ent = np.array([int(x * escala) for x in iota_frac], dtype=np.int64)

        maximo = max(abs(int(x * escala)) for x in todos)
        if maximo * (self.k + 1) ** 2 >= _COTA_ENTEROS:
            raise OverflowError(
                "coeficientes demasiado grandes para la aritmética entera exacta "
                f"(escala={escala}, max={maximo}, k={self.k})"
            )

    @property
    def H_extendida(self) -> list:
        """H en la escala original como racionales, n×(n+1)."""
        return [[Fraction(int(x), self.escala) for x in f] for f in self.H_ent]

    def parametros(self) -> dict:
        """Parámetros en forma serializable (escala original)."""
        return {
            "L": self.L,
            "poblaciones": list(self.poblaciones),
            "H": [[float(x) for x in f] for f in self.H_extendida],
            "tau": [float(Fraction(int(x), self.escala)) for x in self.tau_ent],
            "iota": [float(Fraction(int(x), self.escala)) for x in self.iota_ent],
            "modelo": self.modelo,
            "vecindad": self.vecindad,
            "radio": self.radio,
            "k": self.k,
            "busqueda": self.busqueda,
            "T_max": self.T_max,
            "seed": self.seed,
            "parar_en_repeticion": self.parar_en_repeticion,
        }


def _vector(valor, n: int, nombre: str) -> list:
    """Escalar o vector de longitud n -> lista de n racionales."""
    if np.ndim(valor) == 0:
        return [a_fraccion(valor)] * n
    lista = list(valor)
    if len(lista) != n:
        raise ValueError(f"{nombre} debe ser escalar o tener longitud {n}")
    return [a_fraccion(x) for x in lista]
