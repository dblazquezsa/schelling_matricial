import numpy as np
import pytest

from schelling_matricial import Config, Simulacion, ejecutar_replicas

CFG = dict(L=20, poblaciones=(180, 180), H=[[0, 1], [1, 0]], tau=0.5, modelo="P")


def comparables(r):
    return {k: v for k, v in r.items() if k != "git"}


def test_misma_semilla_mismo_resultado():
    s1, s2 = Simulacion(Config(**CFG, seed=42)), Simulacion(Config(**CFG, seed=42))
    r1, r2 = s1.ejecutar(), s2.ejecutar()
    assert np.array_equal(s1.reticula, s2.reticula)
    for clave, valor in r1.items():
        np.testing.assert_equal(valor, r2[clave])


def test_semillas_distintas_resultados_distintos():
    s1, s2 = Simulacion(Config(**CFG, seed=1)), Simulacion(Config(**CFG, seed=2))
    s1.ejecutar(), s2.ejecutar()
    assert not np.array_equal(s1.reticula, s2.reticula)


def test_replicas_y_csv(tmp_path):
    ruta = tmp_path / "resultados.csv"
    filas = ejecutar_replicas(Config(**CFG, T_max=50), semillas=range(3), salida=ruta)
    assert [f["seed"] for f in filas] == [0, 1, 2]
    assert "C_occ_1_2" in filas[0] and "C_full_2_vac" in filas[0] and "H_1_vac" in filas[0]
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 4
    assert filas == ejecutar_replicas(Config(**CFG, T_max=50), semillas=range(3))


def test_busqueda_no_implementada():
    with pytest.raises(NotImplementedError):
        Simulacion(Config(**CFG, busqueda="mejor"))


def test_config_valida():
    with pytest.raises(ValueError):
        Config(L=5, poblaciones=(20, 10), H=[[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        Config(L=5, poblaciones=(2, 2), H=[[0, 1, 2, 3], [1, 0, 0, 0]])
    with pytest.raises(ValueError):
        Config(L=5, poblaciones=(2, 2), H=[[0, 1], [1, 0]], modelo="Q")
    with pytest.raises(ValueError):
        Config(L=5, poblaciones=(2, 2), H=[[0, 1], [1, 0]], tau=[0.1, 0.2, 0.3])
