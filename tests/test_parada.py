from schelling_matricial import Config, Simulacion


def test_satisfecho():
    cfg = Config(L=10, poblaciones=(40, 40), H=[[0, 1], [1, 0]], tau=1, seed=0)
    r = Simulacion(cfg).ejecutar()
    assert (r["motivo_parada"], r["tipo_punto_fijo"]) == ("punto_fijo", "satisfecho")
    assert r["barridos"] == 1 and r["movimientos_totales"] == 0


def test_satisfecho_tras_moverse():
    cfg = Config(L=20, poblaciones=(150, 150), H=[[0, 1], [1, 0]], tau=0.5, seed=0)
    r = Simulacion(cfg).ejecutar()
    assert (r["motivo_parada"], r["tipo_punto_fijo"]) == ("punto_fijo", "satisfecho")
    assert r["movimientos_totales"] > 0 and all(r["u"] == 0)


def test_bloqueado():
    # Un solo tipo, siempre incómodo (I = 0 > tau = -1) y ningún vacío mejora.
    cfg = Config(L=5, poblaciones=(20,), H=[[0]], tau=-1, seed=0)
    r = Simulacion(cfg).ejecutar()
    assert (r["motivo_parada"], r["tipo_punto_fijo"]) == ("punto_fijo", "bloqueado")
    assert r["u"][0] == 1


def test_t_max():
    cfg = Config(L=20, poblaciones=(150, 150), H=[[0, 1], [1, 0]], tau=0.3, seed=0, T_max=1)
    r = Simulacion(cfg).ejecutar()
    assert r["motivo_parada"] == "t_max" and r["tipo_punto_fijo"] is None
    assert r["barridos"] == 1 and r["movimientos_totales"] > 0


def test_estado_repetido():
    # Persecución: el tipo 1 busca al 2 y el 2 huye del 1. En un toro pequeño
    # la dinámica no se detiene y acaba repitiendo estados.
    comun = dict(L=5, poblaciones=(1, 1), H=[[0, -1], [1, 0]], tau=(-1, 0), modelo="S",
                 seed=0, T_max=2000)
    r = Simulacion(Config(**comun)).ejecutar()
    assert r["motivo_parada"] == "estado_repetido" and r["repeticion_detectada"]
    r = Simulacion(Config(**comun, parar_en_repeticion=False)).ejecutar()
    assert r["motivo_parada"] == "t_max" and r["repeticion_detectada"]
    assert r["barridos"] == 2000


def test_serie_temporal():
    cfg = Config(L=20, poblaciones=(150, 150), H=[[0, 1], [1, 0]], tau=0.5, seed=0, cada=2)
    r = Simulacion(cfg).ejecutar()
    assert [e["barrido"] for e in r["serie"]] == list(range(0, r["barridos"] + 1, 2))
