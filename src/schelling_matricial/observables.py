"""Incomodidad exacta sobre toda la retícula y observables de un estado."""

import numpy as np

from .geometria import contar_vecinos, desplazamientos


def campo_incomodidad(cuentas: np.ndarray, tipo: int, cfg):
    """Incomodidad exacta que tendría un agente de tipo `tipo` (0..n-1) en cada casilla.

    Devuelve (num, den), arrays enteros L×L en la escala entera de `cfg`, con
    I = num / (den * escala):
    - Modelo S: num = Σ_j n_j H[i,j] + n_vac H[i,vac], den = 1.
    - Modelo P: num = Σ_j n_j H[i,j], den = n; en casillas aisladas (n = 0)
      se usa (iota, 1).

    Las cuentas no incluyen la propia casilla, así que en la posición de un
    agente el resultado es su incomodidad real.
    """
    H = cfg.H_ent[tipo]
    n = cfg.n
    suma_ocupados = np.tensordot(H[:n], cuentas, axes=1)
    ocupados = cuentas.sum(axis=0)
    if cfg.modelo == "S":
        num = suma_ocupados + (cfg.k - ocupados) * H[n]
        den = np.ones_like(num)
    else:
        aislado = ocupados == 0
        num = np.where(aislado, cfg.iota_ent[tipo], suma_ocupados)
        den = np.where(aislado, 1, ocupados)
    return num, den


def calcular_observables(reticula: np.ndarray, cfg, cuentas: np.ndarray = None) -> dict:
    """Observables de un estado (sección 2 de la especificación).

    Los índices de tipo en los arrays empiezan en 0 (tipo 1 -> índice 0). En
    `C_full` la última columna es el vacío.
    """
    n, k = cfg.n, cfg.k
    if cuentas is None:
        cuentas = contar_vecinos(reticula, n, desplazamientos(cfg.vecindad, cfg.radio))
    ocupados = cuentas.sum(axis=0)

    N_tipo = np.array([np.count_nonzero(reticula == i + 1) for i in range(n)])
    N = int(N_tipo.sum())

    u = np.zeros(n)
    I_media = np.zeros(n)
    C_occ = np.zeros((n, n))
    C_full = np.zeros((n, n + 1))
    aislados = np.zeros(n, dtype=np.int64)
    aristas_ocupadas = 0  # cada arista ocupada se cuenta dos veces
    aristas_distintas = 0

    for i in range(n):
        mascara = reticula == i + 1
        if not mascara.any():
            u[i] = I_media[i] = np.nan
            C_occ[i] = C_full[i] = np.nan
            continue
        num, den = campo_incomodidad(cuentas, i, cfg)
        num, den = num[mascara], den[mascara]
        u[i] = np.mean(num > cfg.tau_ent[i] * den)
        I_media[i] = np.mean(num / den) / cfg.escala

        c = cuentas[:, mascara]  # (n, N_i)
        n_occ = ocupados[mascara]
        aislados[i] = np.count_nonzero(n_occ == 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(n_occ > 0, c / np.maximum(n_occ, 1), 0.0)
        C_occ[i] = frac.mean(axis=1)
        C_full[i, :n] = (c / k).mean(axis=1)
        C_full[i, n] = ((k - n_occ) / k).mean()

        aristas_ocupadas += int(n_occ.sum())
        aristas_distintas += int((n_occ - c[i]).sum())

    pesos = N_tipo / N if N > 0 else np.zeros(n)
    h = float(np.nansum(pesos * np.diag(C_occ)))
    rho_int = aristas_distintas / aristas_ocupadas if aristas_ocupadas > 0 else np.nan
    rho_int_azar = 1.0 - float(np.sum(pesos**2))
    s = 1.0 - rho_int / rho_int_azar if rho_int_azar > 0 else np.nan

    return {
        "u": u,
        "I_media": I_media,
        "C_occ": C_occ,
        "C_full": C_full,
        "aislados": aislados,
        "frac_aislados": float(aislados.sum() / N) if N > 0 else np.nan,
        "h": h,
        "rho_int": rho_int,
        "rho_int_azar": rho_int_azar,
        "s": s,
    }


def phi(reticula: np.ndarray, S, vecindad: str = "moore", radio: int = 1):
    """Potencial de aristas Φ = Σ_{aristas {u,v} ocupadas} S[t(u), t(v)].

    `S` debe ser simétrica (n×n, índices de tipo desde 0). Con `S` entera el
    resultado es un entero exacto.
    """
    S = np.asarray(S)
    total = 0
    for dx, dy in desplazamientos(vecindad, radio):
        # Cada arista no dirigida una sola vez: la mitad "positiva" de los desplazamientos.
        if dx < 0 or (dx == 0 and dy < 0):
            continue
        vecino = np.roll(reticula, shift=(-dx, -dy), axis=(0, 1))
        ambos = (reticula > 0) & (vecino > 0)
        total = total + S[reticula[ambos] - 1, vecino[ambos] - 1].sum()
    return total.item() if hasattr(total, "item") else total
