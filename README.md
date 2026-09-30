# schelling_matricial

Extensión matricial del modelo de desplazamiento de Schelling: motor en `numpy`
para los modelos P (promediado) y S (de suma), con `n` tipos, matriz de
afinidad–aversión `H`, retícula toroidal y vecindades de Moore / von Neumann.
La especificación de esta versión (0.1) está en
[`docs/especificacion_fase0.md`](docs/especificacion_fase0.md) y la guía
completa de uso, en el [manual de usuario](docs/manual_usuario.md).

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Uso mínimo

```python
from schelling_matricial import Config, Simulacion, ejecutar_replicas

cfg = Config(
    L=40, poblaciones=(700, 700),   # N_1, N_2; el resto son vacíos
    H=[[0, 1], [1, 0]],             # n×n o n×(n+1) (última columna: vacío)
    tau=0.5, iota=0, modelo="P",
    vecindad="moore", radio=1,
    busqueda="cercano", T_max=1000, seed=123,
)
sim = Simulacion(cfg)
resultado = sim.ejecutar()          # dict: observables finales + metadatos
print(resultado["motivo_parada"], resultado["tipo_punto_fijo"], resultado["s"])
sim.reticula                        # np.ndarray L×L (0 = vacío, 1..n tipos)

# Réplicas: una fila por semilla, con CSV opcional
filas = ejecutar_replicas(cfg, semillas=range(50), salida="resultados/clasico.csv")
```

Las decisiones de los agentes usan aritmética entera exacta (ver
`config.py`), así que el resultado no depende de errores de redondeo.

## Pruebas

```bash
pytest
```

## Estructura

- `src/schelling_matricial/`: `config.py` (validación y enteros exactos),
  `geometria.py`, `motor.py` (simulación), `observables.py`,
  `experimentos.py` (réplicas y CSV).
- `tests/`: criterios de aceptación de la especificación.
- `referencia/`: código Mesa antiguo, solo lectura (el paquete no lo importa).
