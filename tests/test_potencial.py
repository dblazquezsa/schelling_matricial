from fractions import Fraction

import numpy as np
import pytest

from schelling_matricial import Config, Simulacion, phi


def construir_H(S, w, H_vac):
    """H[i,j] = w_i S[i,j] + H[i,vac]."""
    n = len(S)
    return [[w[i] * S[i][j] + H_vac[i] for j in range(n)] + [H_vac[i]] for i in range(n)]


CASOS = {
    "clasico_suma": dict(S=[[0, 1], [1, 0]], w=(1, 1), H_vac=(0, 0), tau=2),
    "pesos_1_3": dict(S=[[0, 1], [1, 0]], w=(1, 3), H_vac=(0, 0), tau=(2, 5)),
    "diagonal_no_nula": dict(S=[[-1, 2], [2, -1]], w=(2, 1), H_vac=(1, -1), tau=(3, -6)),
    "tres_tipos": dict(S=[[-1, 1, 0], [1, 0, 2], [0, 2, -2]], w=(1, 2, 3),
                       H_vac=(0, 1, 0), tau=(0, 3, -1)),
}


@pytest.mark.parametrize("nombre", CASOS)
def test_potencial_decrece_en_cada_movimiento(nombre):
    caso = CASOS[nombre]
    n = len(caso["S"])
    H = construir_H(caso["S"], caso["w"], caso["H_vac"])
    for seed in range(3):
        cfg = Config(L=10, poblaciones=(25,) * n, H=H, tau=caso["tau"], modelo="S",
                     seed=seed, T_max=200, parar_en_repeticion=False)
        sim = Simulacion(cfg)
        estado = {"phi": phi(sim.reticula, caso["S"]), "movimientos": 0}

        def al_mover(a, tipo, x, y, I_x, I_y):
            nuevo = phi(sim.reticula, caso["S"])
            delta = nuevo - estado["phi"]
            assert Fraction(delta) == (I_y - I_x) / caso["w"][tipo - 1]
            assert delta < 0
            estado["phi"] = nuevo
            estado["movimientos"] += 1

        resultado = sim.ejecutar(al_mover=al_mover)
        assert estado["movimientos"] > 0
        # Con potencial, la dinámica S siempre llega a un punto fijo
        assert resultado["motivo_parada"] == "punto_fijo"


def test_phi_clasico_cuenta_aristas_distintas():
    ret = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 0]])
    # En toro 3x3 Moore todas las casillas son vecinas: aristas (1,2),(1,1),(2,1)
    assert phi(ret, [[0, 1], [1, 0]]) == 2
