from fractions import Fraction

import pytest

from schelling_matricial import Config, Simulacion

H = [[Fraction(1, 3), 2, -1], [0, Fraction(5, 7), 1], [3, -2, Fraction(1, 2)]]


@pytest.mark.parametrize("vecindad, radio", [("moore", 1), ("von_neumann", 2), ("moore", 2)])
def test_I_S_igual_a_k_por_I_P_sin_vacios(vecindad, radio):
    L = 10
    comun = dict(L=L, poblaciones=(30, 30, 40), H=H, vecindad=vecindad, radio=radio, seed=2)
    sim_P = Simulacion(Config(modelo="P", **comun))
    sim_S = Simulacion(Config(modelo="S", **comun))
    k = sim_S.cfg.k
    assert (sim_P.reticula == sim_S.reticula).all()
    assert (sim_P.reticula > 0).all()
    for I_P, I_S in zip(sim_P.incomodidades(), sim_S.incomodidades()):
        assert I_S == k * I_P
