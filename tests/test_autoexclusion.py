from fractions import Fraction

import numpy as np

from schelling_matricial import Config, Simulacion


def reticula_con(L, agentes):
    ret = np.zeros((L, L), dtype=int)
    for (f, c), t in agentes.items():
        ret[f, c] = t
    return ret


def test_incomodidad_en_destino_adyacente_sin_el_propio_agente():
    # Tipo 1 incómodo con los de su tipo (H[0,0] = 1). El agente A está en (3,4)
    # y B, del mismo tipo, en (3,3). El vacío (3,5) es vecino de A pero no de B.
    ret = reticula_con(7, {(3, 3): 1, (3, 4): 1})
    cfg = Config(L=7, poblaciones=(2, 0), H=[[1, 0], [0, 0]], tau=0, modelo="S", seed=0)
    sim = Simulacion(cfg, reticula=ret)
    assert sim.incomodidad((3, 4)) == 1
    assert sim.incomodidad((3, 5), tipo=1, excluir=(3, 4)) == 0
    # Sin excluir contaría al propio agente, que es lo que hacía el código antiguo
    assert sim.incomodidad((3, 5), tipo=1) == 1


def test_el_agente_se_mueve_a_un_destino_adyacente():
    # Con autoconteo ningún vecino de A mejoraría y tendría que ir a distancia 2.
    ret = reticula_con(7, {(3, 3): 1, (3, 4): 1})
    for seed in range(20):
        cfg = Config(L=7, poblaciones=(2, 0), H=[[1, 0], [0, 0]], tau=0, modelo="S", seed=seed)
        sim = Simulacion(cfg, reticula=ret)
        movimientos = []
        sim.barrido(al_mover=lambda a, t, x, y, Ix, Iy: movimientos.append((x, y, Ix, Iy)))
        assert len(movimientos) == 1
        (x, y, Ix, Iy), = movimientos
        d = max(min(abs(x[0] - y[0]) % 7, 7 - abs(x[0] - y[0]) % 7),
                min(abs(x[1] - y[1]) % 7, 7 - abs(x[1] - y[1]) % 7))
        assert d == 1
        assert (Ix, Iy) == (1, 0)


def test_autoexclusion_en_modelo_P():
    # En P el destino adyacente tiene como único vecino ocupado al propio
    # agente: sin él está aislado y vale iota.
    ret = reticula_con(7, {(3, 3): 1, (3, 4): 2})
    cfg = Config(L=7, poblaciones=(1, 1), H=[[2, 5], [5, 3]], tau=0, iota=Fraction(1, 3),
                 modelo="P", seed=0)
    sim = Simulacion(cfg, reticula=ret)
    assert sim.incomodidad((3, 3)) == 5
    assert sim.incomodidad((3, 2), tipo=1, excluir=(3, 3)) == Fraction(1, 3)
    assert sim.incomodidad((2, 4), tipo=1, excluir=(3, 3)) == 5
