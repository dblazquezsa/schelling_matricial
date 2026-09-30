import numpy as np
import pytest

from schelling_matricial import Config, Simulacion
from schelling_matricial.geometria import contar_vecinos, desplazamientos, numero_vecinos


@pytest.mark.parametrize("vecindad, radio, k", [
    ("moore", 1, 8), ("moore", 2, 24), ("moore", 3, 48),
    ("von_neumann", 1, 4), ("von_neumann", 2, 12), ("von_neumann", 3, 24),
])
def test_numero_de_vecinos(vecindad, radio, k):
    assert numero_vecinos(vecindad, radio) == k
    d = desplazamientos(vecindad, radio)
    assert len(d) == k
    assert len({tuple(x) for x in d}) == k
    assert (0, 0) not in {tuple(x) for x in d}


def test_L_demasiado_pequeno():
    with pytest.raises(ValueError):
        Config(L=4, poblaciones=(3, 3), H=[[0, 1], [1, 0]], radio=2)


@pytest.mark.parametrize("vecindad, radio", [("moore", 1), ("von_neumann", 2), ("moore", 2)])
def test_cuentas_incrementales_tras_1000_movimientos(vecindad, radio):
    cfg = Config(L=12, poblaciones=(40, 40, 30), H=np.zeros((3, 3)),
                 vecindad=vecindad, radio=radio, seed=7)
    sim = Simulacion(cfg)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        a = rng.integers(len(sim.tipos))
        j = rng.integers(len(sim._vac_f))
        sim._mover(a, j)
    desde_cero = contar_vecinos(sim.reticula, cfg.n, desplazamientos(vecindad, radio))
    assert np.array_equal(sim._cuentas, desde_cero)
    # La lista de vacíos y las posiciones siguen siendo coherentes
    assert np.all(sim.reticula[sim._vac_f, sim._vac_c] == 0)
    assert np.array_equal(sim.reticula[sim._pos_f, sim._pos_c], sim.tipos)


def test_cuentas_incrementales_en_la_dinamica():
    cfg = Config(L=15, poblaciones=(90, 90), H=[[0, 1], [1, 0]], tau=0.3, seed=3)
    sim = Simulacion(cfg)
    for _ in range(5):
        sim.barrido()
    assert sim.movimientos_totales > 0
    assert np.array_equal(sim._cuentas, contar_vecinos(sim.reticula, 2, desplazamientos("moore", 1)))
