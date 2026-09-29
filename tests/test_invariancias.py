from fractions import Fraction

import numpy as np

from conftest import iguales, trayectoria
from schelling_matricial import Config

H2 = [[0, 1], [1, 0]]
H3 = [[Fraction(-1, 2), 1, Fraction(3, 10)], [2, 0, Fraction(1, 5)], [1, Fraction(7, 10), -1]]
BASE_P = dict(L=12, modelo="P", seed=5)


def transformar(H, lam, mu, columnas_vac=True):
    H = np.array([[Fraction(x) for x in f] for f in H], dtype=object)
    if H.shape[1] == H.shape[0]:
        H = np.hstack([H, np.full((H.shape[0], 1), Fraction(0), dtype=object)])
    Hp = H * lam
    if columnas_vac:
        Hp = Hp + mu
    else:
        Hp[:, :-1] = Hp[:, :-1] + mu
    return Hp.tolist()


def casos_P():
    yield dict(poblaciones=(50, 50), H=H2, tau=Fraction(1, 2), iota=0)
    yield dict(poblaciones=(40, 40, 30), H=H3, tau=Fraction(3, 5), iota=Fraction(1, 4))
    yield dict(poblaciones=(15, 15), H=H2, tau=Fraction(1, 3), iota=Fraction(1, 5))


def test_invariancia_P():
    for caso in casos_P():
        for lam, mu in [(Fraction(5, 2), Fraction(-7, 10)), (Fraction(1, 3), 4)]:
            original = Config(**BASE_P, **caso)
            transformado = Config(
                **BASE_P, poblaciones=caso["poblaciones"],
                H=transformar(caso["H"], lam, mu),
                tau=lam * caso["tau"] + mu, iota=lam * caso["iota"] + mu,
            )
            t1, t2 = trayectoria(original), trayectoria(transformado)
            assert len(t1) > 2
            assert iguales(t1, t2)


def test_no_invariancia_de_iota_en_P():
    # Baja densidad: hay vacíos aislados disponibles. Sin transformar iota, la
    # casilla aislada cambia de valor relativo (observación O3). Con λ=1, μ=-1,
    # iota'=0 equivale a una fracción de distintos f=1 en la escala original.
    lam, mu = Fraction(1), Fraction(-1)
    distintas = 0
    for seed in range(5):
        base = dict(L=12, poblaciones=(15, 15), modelo="P", seed=seed)
        original = Config(**base, H=H2, tau=Fraction(1, 2), iota=0)
        sin_iota = Config(**base, H=transformar(H2, lam, mu), tau=lam * Fraction(1, 2) + mu, iota=0)
        distintas += not iguales(trayectoria(original), trayectoria(sin_iota))
    assert distintas > 0


def casos_S():
    yield dict(poblaciones=(50, 50), H=H2, tau=2)
    yield dict(poblaciones=(40, 40, 30), H=[f + [Fraction(1, 10)] for f in H3], tau=Fraction(3, 2))


def test_invariancia_S():
    k = 8
    for caso in casos_S():
        for lam, mu in [(Fraction(5, 2), Fraction(-7, 10)), (Fraction(1, 3), 4)]:
            original = Config(L=12, modelo="S", seed=5, **caso)
            transformado = Config(
                L=12, modelo="S", seed=5, poblaciones=caso["poblaciones"],
                H=transformar(caso["H"], lam, mu), tau=lam * caso["tau"] + k * mu,
            )
            t1, t2 = trayectoria(original), trayectoria(transformado)
            assert len(t1) > 2
            assert iguales(t1, t2)


def test_sumar_solo_a_ocupados_cambia_S():
    k, lam, mu = 8, Fraction(1), Fraction(1, 2)
    distintas = 0
    for seed in range(5):
        base = dict(L=12, poblaciones=(40, 40), modelo="S", seed=seed)
        original = Config(**base, H=H2, tau=2)
        solo_ocupados = Config(**base, H=transformar(H2, lam, mu, columnas_vac=False),
                               tau=lam * 2 + k * mu)
        distintas += not iguales(trayectoria(original), trayectoria(solo_ocupados))
    assert distintas > 0
