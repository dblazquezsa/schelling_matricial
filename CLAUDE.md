# Repositorio schelling_matricial — Extensión matricial del modelo de Schelling

## Contexto
Investigación sobre generalizaciones del modelo de desplazamiento de Schelling.
La dinámica depende de una matriz de afinidad–aversión `H` y de umbrales de
tolerancia `tau`. Hay dos versiones del modelo:

- **Modelo P (promediado):** la incomodidad es el promedio de `H[i, j]` sobre los
  vecinos ocupados, así que depende de proporciones.
- **Modelo S (de suma):** la incomodidad es la suma de `H[i, j]` sobre la
  vecindad, incluidas las casillas vacías con peso `H[i, vacío]`, así que
  depende de recuentos. Es la versión que implementa el código antiguo
  (repositorio `mesa-python`).

Documentos de referencia, que hay que leer antes de trabajar en el paquete:
- `docs/especificacion_fase0.md`: especificación técnica de la primera versión
  del paquete. **Es la fuente de verdad para la Fase 0.**
- `docs/programa_investigacion_schelling_matricial.tex`: programa de
  investigación completo. Las secciones 3 y 4 contienen el marco matemático,
  los observables y el protocolo.

## Estructura del repositorio
- `src/schelling_matricial/`: código del paquete (motor numpy).
- `tests/`: pruebas con `pytest`.
- `docs/`: especificación y programa de investigación.
- `referencia/`: copia **de solo lectura** del código Mesa antiguo del
  repositorio `mesa-python` (`modelo.py`, `schelling_general2.py`,
  `schelling_general3.py`). Sirve para consultar el comportamiento anterior y
  para la validación cualitativa. `modelo.py` es el modelo de tres grupos con
  población neutral. **No modificar ni importar desde el paquete.**

## Convenciones
- Python ≥ 3.10 y `numpy`; `pytest` para las pruebas; `mesa` solo para una
  visualización opcional futura (no es dependencia del paquete).
- Nombres, comentarios y docstrings en español, como en el código antiguo.
- Aritmética exacta en las decisiones de los agentes: se trabaja con enteros
  (ver la especificación). Nunca se comparan promedios en coma flotante.
- Toda simulación recibe una semilla y es reproducible.
- Una tarea no está terminada hasta que `pytest` pasa completo.
- Commits pequeños y descriptivos, uno por módulo o funcionalidad.

## Forma de trabajar
- Trabajamos por fases pequeñas. Ante una decisión de modelado que la
  especificación no cubra, pregunta en lugar de suponer.
- Explica los cambios de forma breve: el responsable está aprendiendo a usar
  Claude Code.
