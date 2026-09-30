from fractions import Fraction

import pytest

from schelling_matricial import Config, Simulacion
from schelling_matricial.geometria import desplazamientos


@pytest.mark.parametrize("vecindad, radio", [("moore", 1), ("von_neumann", 2)])
def test_incomodidad_es_fraccion_de_distintos(vecindad, radio):
    cfg = Config(L=9, poblaciones=(25, 25), H=[[0, 1], [1, 0]], modelo="P",
                 vecindad=vecindad, radio=radio, seed=11)
    sim = Simulacion(cfg)
    ret, L = sim.reticula, cfg.L
    for (f, c), I in zip(sim.posiciones, sim.incomodidades()):
        propio = ret[f, c]
        vecinos = [ret[(f + dx) % L, (c + dy) % L] for dx, dy in desplazamientos(vecindad, radio)]
        ocupados = [v for v in vecinos if v != 0]
        distintos = sum(1 for v in ocupados if v != propio)
        esperado = Fraction(distintos, len(ocupados)) if ocupados else Fraction(0)
        assert I == esperado
