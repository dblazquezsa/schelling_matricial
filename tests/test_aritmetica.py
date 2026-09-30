from fractions import Fraction
from itertools import permutations

import numpy as np

from schelling_matricial import Config, Simulacion
from schelling_matricial.geometria import desplazamientos


def test_agente_exactamente_en_el_umbral_es_comodo():
    cfg = Config(L=5, poblaciones=(5, 4), H=[[0.4, 0.6], [0.6, 0.4]], tau=0.5, modelo="P", seed=0)
    desp = desplazamientos("moore", 1)
    vistos = set()
    for orden in permutations([1, 1, 1, 1, 2, 2, 2, 2]):
        if orden in vistos:
            continue
        vistos.add(orden)
        ret = np.zeros((5, 5), dtype=int)
        ret[2, 2] = 1
        for (dx, dy), t in zip(desp, orden):
            ret[2 + dx, 2 + dy] = t
        sim = Simulacion(cfg, reticula=ret)
        centro = int(np.flatnonzero((sim.posiciones == [2, 2]).all(axis=1))[0])
        assert sim.incomodidad((2, 2)) == Fraction(1, 2)
        assert sim.comodos()[centro]
    assert len(vistos) == 70


def test_conversion_a_enteros():
    cfg = Config(L=5, poblaciones=(1, 1), H=[[0.1, 1 / 3], [0.25, 0]], tau=0.3, iota=0.2)
    # mcm(10, 3, 4, 10, 5) = 60
    assert cfg.escala == 60
    assert cfg.H_ent.tolist() == [[6, 20, 0], [15, 0, 0]]
    assert cfg.tau_ent.tolist() == [18, 18]
    assert cfg.iota_ent.tolist() == [12, 12]
    assert cfg.H_ent.dtype == np.int64
