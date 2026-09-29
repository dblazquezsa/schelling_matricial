"""Motor de simulación: estado, campo de cuentas, barrido asíncrono y parada."""

import hashlib
import subprocess
from fractions import Fraction
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .config import Config
from .geometria import contar_vecinos, desplazamientos, distancia_chebyshev, son_vecinos
from .observables import calcular_observables

try:
    __version__ = version("schelling_matricial")
except PackageNotFoundError:  # paquete sin instalar
    __version__ = "0.1.0"

REGLAS_IMPLEMENTADAS = ("cercano",)


@lru_cache(maxsize=1)
def hash_git():
    """Hash del commit de git del código, o None si no se puede obtener."""
    try:
        salida = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return salida.stdout.strip() or None if salida.returncode == 0 else None


class Simulacion:
    """Una simulación de los modelos P / S sobre la retícula toroidal.

    Estado:
    - `reticula`: array L×L con 0 (vacío) o el tipo 1..n.
    - `_cuentas`: campo de recuentos (n, L, L); `_cuentas[j, f, c]` es el número
      de vecinos de tipo j+1 de la casilla (f, c). Se actualiza localmente en
      cada movimiento.
    - `_vac_f`, `_vac_c`: coordenadas de las casillas vacías.
    - `_pos_f`, `_pos_c`, `_tipo`: posición y tipo (0..n-1) de cada agente.

    Si se pasa `reticula`, se usa como estado inicial (sus poblaciones deben
    coincidir con las de `cfg`); si no, los agentes se colocan al azar.
    """

    def __init__(self, cfg: Config, reticula: np.ndarray = None):
        if cfg.busqueda not in REGLAS_IMPLEMENTADAS:
            raise NotImplementedError(
                f"la regla de búsqueda {cfg.busqueda!r} aún no está implementada "
                f"(disponibles: {REGLAS_IMPLEMENTADAS})"
            )
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        L, n = cfg.L, cfg.n

        if reticula is None:
            N = sum(cfg.poblaciones)
            casillas = self.rng.choice(L * L, size=N, replace=False)
            tipos = np.repeat(np.arange(1, n + 1), cfg.poblaciones)
            self.reticula = np.zeros(L * L, dtype=np.int64)
            self.reticula[casillas] = tipos
            self.reticula = self.reticula.reshape(L, L)
        else:
            self.reticula = np.array(reticula, dtype=np.int64)
            if self.reticula.shape != (L, L):
                raise ValueError(f"la retícula debe tener forma {(L, L)}")
            if self.reticula.min() < 0 or self.reticula.max() > n:
                raise ValueError(f"los valores de la retícula deben estar en 0..{n}")
            recuento = tuple(int(np.count_nonzero(self.reticula == i + 1)) for i in range(n))
            if recuento != cfg.poblaciones:
                raise ValueError(f"la retícula tiene poblaciones {recuento}, no {cfg.poblaciones}")

        # Agentes: en el orden de la retícula (fila, columna).
        self._pos_f, self._pos_c = np.nonzero(self.reticula)
        self._tipo = self.reticula[self._pos_f, self._pos_c] - 1
        self._vac_f, self._vac_c = np.nonzero(self.reticula == 0)

        self._desp = desplazamientos(cfg.vecindad, cfg.radio)
        self._cuentas = contar_vecinos(self.reticula, n, self._desp)

        # Coeficientes enteros, como listas/arrays listos para usar.
        H = cfg.H_ent
        self._H_occ = H[:, :n]                    # parte de tipos ocupados
        self._H_occ_rel = H[:, :n] - H[:, [n]]    # modelo S: H[i,j] - H[i,vac]
        self._base_S = [int(cfg.k * H[i, n]) for i in range(n)]  # k·H[i,vac]
        self._tau = [int(x) for x in cfg.tau_ent]
        self._iota = [int(x) for x in cfg.iota_ent]

        self.barridos = 0
        self.movimientos_totales = 0
        self._hashes = {self._hash_estado()}
        self.barrido_repeticion = None

    # ------------------------------------------------------------------
    # Incomodidad
    # ------------------------------------------------------------------
    def _incomodidad_entera(self, tipo: int, cuentas_x) -> tuple:
        """(num, den) enteros para un agente de tipo `tipo` con esas cuentas de vecinos."""
        if self.cfg.modelo == "S":
            return int(self._H_occ_rel[tipo] @ cuentas_x) + self._base_S[tipo], 1
        den = int(cuentas_x.sum())
        if den == 0:
            return self._iota[tipo], 1
        return int(self._H_occ[tipo] @ cuentas_x), den

    def _comodo(self, tipo: int, num: int, den: int) -> bool:
        return num <= self._tau[tipo] * den

    def _a_fraccion(self, num: int, den: int) -> Fraction:
        """Pasa (num, den) de la escala entera a la escala original de H."""
        return Fraction(num, den * self.cfg.escala)

    def incomodidad(self, pos, tipo: int = None, excluir=None) -> Fraction:
        """Incomodidad exacta (escala original) de un agente de tipo `tipo` (1..n) en `pos`.

        Si `tipo` es None se usa el del agente que ocupa `pos`. Si `excluir` es
        una casilla, se trata como vacía (autoexclusión del agente que se mueve).
        """
        f, c = pos
        if tipo is None:
            tipo = int(self.reticula[f, c])
            if tipo == 0:
                raise ValueError("la casilla está vacía: indica el tipo")
        cuentas_x = self._cuentas[:, f, c].copy()
        if excluir is not None:
            ef, ec = excluir
            t_ex = int(self.reticula[ef, ec])
            if t_ex > 0 and son_vecinos(f, c, ef, ec, self.cfg.L, self.cfg.vecindad, self.cfg.radio):
                cuentas_x[t_ex - 1] -= 1
        return self._a_fraccion(*self._incomodidad_entera(tipo - 1, cuentas_x))

    def incomodidades(self) -> list:
        """Incomodidad exacta de cada agente (en el orden de `posiciones`)."""
        return [
            self._a_fraccion(*self._incomodidad_entera(t, self._cuentas[:, f, c]))
            for f, c, t in zip(self._pos_f, self._pos_c, self._tipo)
        ]

    def comodos(self) -> np.ndarray:
        """Máscara booleana: qué agentes están cómodos."""
        return np.array([
            self._comodo(t, *self._incomodidad_entera(t, self._cuentas[:, f, c]))
            for f, c, t in zip(self._pos_f, self._pos_c, self._tipo)
        ], dtype=bool)

    @property
    def posiciones(self) -> np.ndarray:
        """Posiciones (fila, columna) de los agentes, forma (N, 2)."""
        return np.column_stack([self._pos_f, self._pos_c])

    @property
    def tipos(self) -> np.ndarray:
        """Tipo (1..n) de cada agente."""
        return self._tipo + 1

    # ------------------------------------------------------------------
    # Dinámica
    # ------------------------------------------------------------------
    def _actuar(self, a: int, al_mover=None) -> bool:
        """Acción de un agente (paso 3 de 1.5). Devuelve True si se ha movido."""
        cfg = self.cfg
        t = int(self._tipo[a])
        f, c = int(self._pos_f[a]), int(self._pos_c[a])
        num_x, den_x = self._incomodidad_entera(t, self._cuentas[:, f, c])
        if self._comodo(t, num_x, den_x) or len(self._vac_f) == 0:
            return False

        # Evaluación vectorizada de todos los vacíos, excluyendo al propio
        # agente: si el vacío es vecino de x, el agente no cuenta allí.
        vf, vc = self._vac_f, self._vac_c
        C = self._cuentas[:, vf, vc]
        C[t] -= son_vecinos(f, c, vf, vc, cfg.L, cfg.vecindad, cfg.radio)
        if cfg.modelo == "S":
            num_y = self._H_occ_rel[t] @ C + self._base_S[t]
            den_y = None
            mejora = num_y < num_x
        else:
            num_y = self._H_occ[t] @ C
            den_y = C.sum(axis=0)
            aislado = den_y == 0
            num_y[aislado] = self._iota[t]
            den_y[aislado] = 1
            mejora = num_y * den_x < num_x * den_y

        candidatos = np.flatnonzero(mejora)
        if len(candidatos) == 0:
            return False
        j = self._elegir(candidatos, f, c)

        if al_mover is not None:
            I_x = self._a_fraccion(num_x, den_x)
            I_y = self._a_fraccion(int(num_y[j]), 1 if den_y is None else int(den_y[j]))
            destino = (int(vf[j]), int(vc[j]))
        self._mover(a, j)
        if al_mover is not None:
            al_mover(a, t + 1, (f, c), destino, I_x, I_y)
        return True

    def _elegir(self, candidatos: np.ndarray, f: int, c: int) -> int:
        """Regla de búsqueda: índice (en la lista de vacíos) del destino elegido.

        Aquí se añadirán "aleatorio" (cualquiera que mejore) y "mejor" (menor
        incomodidad, desempate al azar).
        """
        if self.cfg.busqueda == "cercano":
            d = distancia_chebyshev(f, c, self._vac_f[candidatos], self._vac_c[candidatos], self.cfg.L)
            cercanos = candidatos[d == d.min()]
            return int(cercanos[self.rng.integers(len(cercanos))])
        raise NotImplementedError(self.cfg.busqueda)

    def _vecinos(self, f: int, c: int):
        L = self.cfg.L
        return (f + self._desp[:, 0]) % L, (c + self._desp[:, 1]) % L

    def _mover(self, a: int, j: int) -> None:
        """Mueve el agente `a` al vacío j-ésimo y actualiza el estado localmente."""
        t = self._tipo[a]
        f, c = self._pos_f[a], self._pos_c[a]
        yf, yc = self._vac_f[j], self._vac_c[j]
        self.reticula[f, c] = 0
        self.reticula[yf, yc] = t + 1
        self._cuentas[t][self._vecinos(f, c)] -= 1
        self._cuentas[t][self._vecinos(yf, yc)] += 1
        self._vac_f[j], self._vac_c[j] = f, c
        self._pos_f[a], self._pos_c[a] = yf, yc

    def barrido(self, al_mover=None) -> int:
        """Un barrido asíncrono: cada agente actúa una vez, en orden aleatorio.

        `al_mover(agente, tipo, x, y, I_x, I_y)`, si se da, se llama tras cada
        movimiento con las incomodidades exactas (escala original) en el origen
        y en el destino. Devuelve el número de movimientos.
        """
        movimientos = 0
        for a in self.rng.permutation(len(self._tipo)):
            movimientos += self._actuar(int(a), al_mover)
        self.barridos += 1
        self.movimientos_totales += movimientos
        return movimientos

    def _hash_estado(self) -> bytes:
        return hashlib.blake2b(self.reticula.tobytes(), digest_size=16).digest()

    def ejecutar(self, al_mover=None) -> dict:
        """Ejecuta barridos hasta una condición de parada y devuelve el resultado.

        Parada, comprobada tras cada barrido y en este orden:
        - "punto_fijo": nadie se ha movido ("satisfecho" o "bloqueado").
        - "estado_repetido": la retícula ya había aparecido (incluido el estado
          inicial). Es indicio de ciclo, no prueba. Solo para si
          `cfg.parar_en_repeticion`; en todo caso se registra el primer barrido
          en que ocurre.
        - "t_max": se han hecho `cfg.T_max` barridos en total.

        `barridos` cuenta todos los barridos hechos, incluido el último (que en
        un punto fijo es el barrido sin movimientos).
        """
        cfg = self.cfg
        serie = []
        if cfg.cada is not None and self.barridos == 0:
            serie.append(self._registro())

        motivo = "t_max"
        while self.barridos < cfg.T_max:
            movimientos = self.barrido(al_mover)
            if cfg.cada is not None and self.barridos % cfg.cada == 0:
                serie.append(self._registro())
            if movimientos == 0:
                motivo = "punto_fijo"
                break
            h = self._hash_estado()
            if h in self._hashes:
                if self.barrido_repeticion is None:
                    self.barrido_repeticion = self.barridos
                if cfg.parar_en_repeticion:
                    motivo = "estado_repetido"
                    break
            self._hashes.add(h)

        obs = calcular_observables(self.reticula, cfg, self._cuentas)
        tipo_punto_fijo = None
        if motivo == "punto_fijo":
            satisfecho = bool(np.all(np.nan_to_num(obs["u"]) == 0))
            tipo_punto_fijo = "satisfecho" if satisfecho else "bloqueado"

        resultado = {
            **cfg.parametros(),
            "version": __version__,
            "git": hash_git(),
            "barridos": self.barridos,
            "movimientos_totales": self.movimientos_totales,
            "motivo_parada": motivo,
            "tipo_punto_fijo": tipo_punto_fijo,
            "repeticion_detectada": self.barrido_repeticion is not None,
            "barrido_repeticion": self.barrido_repeticion,
            **obs,
        }
        if cfg.cada is not None:
            resultado["serie"] = serie
        return resultado

    def _registro(self) -> dict:
        return {
            "barrido": self.barridos,
            "movimientos_totales": self.movimientos_totales,
            **calcular_observables(self.reticula, self.cfg, self._cuentas),
        }
