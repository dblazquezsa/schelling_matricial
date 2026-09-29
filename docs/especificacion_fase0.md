# Especificación Fase 0 — paquete `schelling_matricial` (versión 0.1)

Este documento es la fuente de verdad para la primera versión del paquete, que
vive en su propio repositorio, `schelling_matricial`. Deriva de las
secciones 3–5 de `docs/programa_investigacion_schelling_matricial.tex`.

## 0. Alcance

**Incluido en v0.1**
- Motor de simulación en `numpy` para los modelos P y S, con `n` tipos,
  retícula toroidal L×L y vecindades Moore / von Neumann de radio `r`.
- Regla de movimiento parametrizada (una regla implementada por ahora).
- Observables básicos y registro del estado final.
- Utilidad de experimentos: réplicas con semillas y salida tabular (CSV).
- Batería de pruebas que verifican las propiedades matemáticas del modelo.

**Fuera de alcance en v0.1** (fases posteriores)
- Redes (Erdős–Rényi), umbrales individuales heterogéneos.
- Barridos grandes de parámetros, análisis de tamaño finito, gráficas de
  resultados.
- Visualización Mesa/Solara: opcional, solo si sobra tiempo, y como
  envoltorio del motor, sin duplicar la lógica.

**No tocar** los ficheros de `referencia/`, que son una copia del código Mesa
antiguo del repositorio `mesa-python`. El paquete no debe importar nada de
esa carpeta.

## 1. Modelo matemático

### 1.1 Espacio y vecindades
- Retícula toroidal `L×L`. Cada casilla vale `0` (vacía) o un tipo `1..n`.
- Vecindad de Moore de radio `r`: desplazamientos `(dx, dy) ≠ (0,0)` con
  `max(|dx|,|dy|) ≤ r`, de modo que `k = (2r+1)² − 1`.
- Vecindad de von Neumann de radio `r`: `(dx, dy) ≠ (0,0)` con
  `|dx|+|dy| ≤ r`, de modo que `k = 2r(r+1)`.
- Se exige `L > 2r` para que no haya vecinos repetidos en el toro. Si no se
  cumple, lanzar un error.
- Para una casilla `x`: `n_j(x)` es el número de vecinos de tipo `j`,
  `n(x) = Σ_j n_j(x)` y `n_vac(x) = k − n(x)`.

### 1.2 Parámetros
- `H`: matriz `n × (n+1)`. Las columnas `0..n−1` corresponden a los tipos
  `1..n` y la última columna al vacío (`H[i, vac]`).
  - Atajo: si se pasa una matriz `n × n`, se añade una columna de ceros.
  - Semántica: `H[i, j]` es la incomodidad que un agente de tipo `i+1` asocia
    a un vecino de tipo `j+1`.
- `tau`: escalar (umbral común) o vector de longitud `n` (umbral por tipo).
- `iota`: escalar o vector de longitud `n`. Es la incomodidad de un agente
  aislado (`n(x) = 0`) en el modelo P. Por defecto vale `0`, que es la
  convención del borrador.
- `modelo`: `"P"` o `"S"`.

### 1.3 Incomodidad
Para un agente de tipo `i` evaluado en la casilla `x`:

- **Modelo S:** `I_S = Σ_j n_j(x)·H[i,j] + n_vac(x)·H[i,vac]`
- **Modelo P:** `I_P = (Σ_j n_j(x)·H[i,j]) / n(x)` si `n(x) > 0`; `I_P = iota[i]`
  si `n(x) = 0`.

**Exclusión del propio agente (importante).** Al evaluar una casilla candidata
`y`, el agente que se mueve no cuenta como vecino de sí mismo. Si `y` es vecina
de su posición actual `x`, `x` debe tratarse como vacía en el cálculo. El código
antiguo tiene este error (el agente se cuenta a sí mismo en destinos
adyacentes) y no debe reproducirse.

### 1.4 Aritmética exacta
Las decisiones (cómodo/incómodo, mejora) deben ser exactas:
- Convertir `H`, `tau` e `iota` a racionales con
  `Fraction(str(x)).limit_denominator(10**6)`, multiplicarlos por el mínimo
  común múltiplo de los denominadores y guardarlos como enteros `int64`.
  Escalar todo por el mismo factor positivo no cambia la dinámica.
- **Modelo S:** `I_S` es un entero. Las comparaciones son directas.
- **Modelo P:** se representa como la fracción `(num, den) = (Σ n_j H[i,j], n)`.
  - Comodidad: `num ≤ tau·den`.
  - Mejora de `x` a `y`: `num_y·den_x < num_x·den_y`, con cuidado cuando
    `den = 0`: en ese caso se usa `iota` (que es un entero, con `den = 1`).
- Documentar el riesgo de desbordamiento, que en la práctica no ocurre con
  `k ≤ 48` y coeficientes moderados.

### 1.5 Dinámica
1. **Inicialización:** `N_i` agentes de cada tipo colocados al azar sin
   repetición. El resto de casillas quedan vacías. Todo sale de
   `numpy.random.default_rng(seed)`.
2. **Barrido:** permutación aleatoria de todos los agentes. Cada agente actúa
   una vez en orden, sobre el estado ya actualizado (actualización asíncrona,
   igual que `shuffle_do` de Mesa).
3. **Acción de un agente** de tipo `i` en `x`:
   - Si `I(x) ≤ tau[i]`, está cómodo y no hace nada.
   - Si está incómodo, construye el conjunto de vacíos que mejoran,
     `{y vacía : I(y) < I(x)}`, evaluados con la exclusión de 1.3.
   - Elige destino según la regla de búsqueda. **v0.1** implementa solo
     `"cercano"`: el subconjunto de mejora con menor distancia de Chebyshev
     toroidal a `x`, y dentro de él un elemento elegido uniformemente al azar.
     Si el conjunto es vacío, no se mueve.
   - Debe dejarse preparada la interfaz para otras reglas (`"aleatorio"`:
     cualquiera que mejore; `"mejor"`: el de menor `I`, desempate al azar),
     aunque no se implementen todavía.
4. **Parada**, tras cada barrido:
   - `"punto_fijo"`: nadie se movió. Se subclasifica como `"satisfecho"` si
     todos están cómodos o `"bloqueado"` si queda algún incómodo.
   - `"estado_repetido"`: el *hash* de la retícula al final del barrido ya
     apareció antes. La dinámica es estocástica, así que esto es un indicio de
     ciclo, no una prueba. Registrarlo y parar solo si
     `parar_en_repeticion=True`, que es el valor por defecto.
   - `"t_max"`: se alcanzó el número máximo de barridos `T_max`.

### 1.6 Eficiencia (requisito de diseño)
- Mantener un campo de recuentos `cuentas[j]` (forma `n × L × L`, entero), con
  `n_j` para todas las casillas. Se calcula al inicio sumando `np.roll` sobre los
  desplazamientos de la vecindad.
- Tras cada movimiento `x → y` de un agente de tipo `t`, actualizarlo
  localmente: restar 1 en `cuentas[t]` en las `k` casillas vecinas de `x` y
  sumar 1 en las de `y`. **No recalcular todo el campo.**
- Mantener un array con las coordenadas de los vacíos para evaluar todos los
  candidatos de forma vectorizada.
- Objetivo orientativo: un barrido con `L = 80`, densidad de vacíos 0,1 y
  Moore `r = 1` en menos de ~1 s en un portátil.

## 2. Observables (mínimo v0.1)
Se calculan sobre un estado, y se registran al final y, opcionalmente, cada
`cada` barridos:
- `u[i]`: fracción de agentes incómodos del tipo `i`.
- `I_media[i]`: incomodidad media del tipo `i`, en la escala original de `H`,
  no en la escala entera.
- `C_occ[i, j]`: media sobre los agentes de tipo `i` de `n_j/n`. Los agentes
  aislados aportan 0 y hay que registrar cuántos son.
- `C_full[i, j]`, con `j` en tipos ∪ {vacío}: media de `n_j/k`.
- `h = Σ_i (N_i/N)·C_occ[i,i]`: homofilia.
- `rho_int`: fracción de aristas con los dos extremos ocupados que unen tipos
  distintos. `s = 1 − rho_int / rho_int_azar`, donde
  `rho_int_azar = 1 − Σ_i (N_i/N)²` es el valor esperado con mezcla aleatoria.
- `frac_aislados`, `barridos`, `movimientos_totales`, `motivo_parada`,
  `tipo_punto_fijo`.
- `phi`: potencial de aristas `Φ = Σ_{aristas ocupadas} S[t(u), t(v)]` para una
  matriz simétrica `S` dada. Es una función auxiliar que usan las pruebas.

## 3. API propuesta

```python
from schelling_matricial import Config, Simulacion, ejecutar_replicas

cfg = Config(
    L=40, poblaciones=(700, 700),                 # N_1, N_2; resto vacíos
    H=[[0, 1], [1, 0]],                           # n×n o n×(n+1)
    tau=0.5, iota=0, modelo="P",
    vecindad="moore", radio=1,
    busqueda="cercano", T_max=1000, seed=123,
)
sim = Simulacion(cfg)
resultado = sim.ejecutar()        # dict: observables finales + metadatos
sim.reticula                      # np.ndarray L×L (0 = vacío, 1..n tipos)

tabla = ejecutar_replicas(cfg, semillas=range(50))   # lista de dicts / DataFrame
```

- `Config` es un `dataclass` validado: dimensiones de `H`, suma de
  poblaciones ≤ L², `L > 2r` y valores permitidos de `modelo`, `vecindad` y
  `busqueda`.
- Cada resultado incluye los parámetros, la semilla, la versión del paquete y,
  si se puede obtener, el *hash* del commit de git.
- `ejecutar_replicas(..., salida="resultados.csv")` escribe un CSV con una fila
  por réplica. Las matrices se aplanan en columnas `C_occ_1_2`, etc.

## 4. Estructura del repositorio

La raíz del repositorio `schelling_matricial` es la raíz del paquete
(disposición `src/`, instalable con `pip install -e .`):

```
schelling_matricial/             # raíz del repositorio
  CLAUDE.md
  README.md                      # uso mínimo y cómo correr las pruebas
  pyproject.toml                 # dependencias: numpy; extras [dev]: pytest
  .gitignore                     # Python estándar + resultados/ + .venv/
  docs/
    especificacion_fase0.md      # este documento
    programa_investigacion_schelling_matricial.tex
  referencia/                    # código Mesa antiguo, solo lectura
    modelo.py
    schelling_general2.py
    schelling_general3.py
  src/schelling_matricial/
    __init__.py
    config.py                    # Config + validación + conversión a enteros exactos
    geometria.py                 # desplazamientos de vecindad, k, distancia toroidal
    motor.py                     # Simulacion: estado, campo de cuentas, barrido, parada
    observables.py
    experimentos.py              # ejecutar_replicas, escritura CSV
  tests/
    test_geometria.py
    test_clasico.py
    test_autoexclusion.py
    test_invariancias.py
    test_potencial.py
    test_P_vs_S.py
    test_aritmetica.py
    test_reproducibilidad.py
  notebooks/validacion_fase0.ipynb   # opcional en v0.1
```

Entorno recomendado: un entorno virtual `.venv` en la raíz
(`python3 -m venv .venv`), con el paquete instalado en modo editable junto con
los extras de desarrollo (`pip install -e ".[dev]"`).

## 5. Pruebas obligatorias (criterios de aceptación)

1. **Geometría.** `k` vale 8, 24 y 48 para Moore con r = 1, 2, 3, y 4, 12 y
   24 para von Neumann con r = 1, 2, 3. El campo de cuentas incremental coincide
   con el recalculado desde cero tras 1000 movimientos aleatorios.
2. **Clásico.** Con `H = [[0,1],[1,0]]` en el modelo P, la incomodidad de cada
   agente es exactamente la fracción de vecinos distintos (se compara con un
   cálculo directo casilla a casilla en una retícula pequeña).
3. **Autoexclusión.** Caso construido a mano: un agente con `H[i,i] ≠ 0` y un
   destino adyacente. La incomodidad calculada en el destino no incluye al
   propio agente.
4. **Invariancia en P.** Con la misma semilla, `H' = λH + μ`, `tau' = λ·tau + μ`
   e `iota' = λ·iota + μ` (λ > 0) producen **la misma trayectoria**, es decir,
   la misma retícula tras cada barrido.
5. **No invariancia de iota en P.** El test anterior, pero sin transformar
   `iota`, debe dar trayectorias distintas en al menos un caso con vacíos
   aislados disponibles. Esto documenta la observación O3 del programa.
6. **Invariancia en S.** Transformando la fila completa, incluida la columna de
   vacío, `H' = λH + μ` con `tau' = λ·tau + k·μ`, la trayectoria es idéntica.
   Sumar μ solo a las columnas ocupadas **cambia** la dinámica en al menos un
   caso.
7. **Potencial en S.** Si `H[i,j] − H[i,vac] = w_i·S[i,j]`, con `S` simétrica
   y `w_i > 0`, entonces en cada movimiento
   `ΔΦ = ΔI_mover / w_i` exactamente, y `Φ` es estrictamente decreciente en
   toda la trayectoria. Probar al menos: el clásico de suma, un caso con
   `w = (1, 3)` y un caso con diagonal no nula.
8. **P = S a densidad máxima.** Sin vacíos (se prueba la igualdad de
   incomodidades, no la dinámica, porque sin vacíos no hay movimiento):
   `I_S = k·I_P` para todos los agentes cuando `H[i,vac] = 0`.
9. **Aritmética en el umbral.** Con `H = [[0.4, 0.6],[0.6, 0.4]]`, `tau = 0.5`
   en el modelo P, un agente con 4 vecinos iguales y 4 distintos está
   **exactamente** en el umbral, por lo que es cómodo, con independencia del
   orden de los vecinos.
10. **Reproducibilidad.** Misma `Config` y misma semilla dan resultados
    idénticos. Semillas distintas dan resultados distintos.
11. **Parada.** Casos pequeños que terminen en `"satisfecho"`, `"bloqueado"` y
    `"t_max"`.

## 6. Validación cualitativa (no automática; informe breve)
- Clásico P con Moore r = 1, densidad de vacíos 0,1 y `tau` en {0,3; 0,5; 0,7}:
  `s` debe crecer con la intolerancia, en línea con la literatura (Gauvin et
  al. 2009).
- Modelo de población neutral: modelo S, 3 tipos, en el orden (1, 0, −1)
  mapeado a (1, 2, 3):
  ```
  H_neu = [[0, 0, 1],
           [1, 0, 1],
           [1, 0, 0]]
  ```
  Con los mismos parámetros que `src/modelo.py`, comparar estadísticamente
  (media ± desviación sobre 25 réplicas) la proporción de contactos del mismo
  tipo y los barridos hasta el equilibrio. Hay que esperar diferencias pequeñas
  por la corrección del autoconteo. Aquí `H[i,i] = 0`, así que no debería haber
  ninguna.

## 7. Decisiones ya tomadas por defecto (modificables)
- `iota = 0` (convención del borrador) y `H[i,vac] = 0` (convención del código
  antiguo).
- `tau` común, aunque se admite un vector por tipo.
- Regla de búsqueda `"cercano"` con distancia de Chebyshev toroidal.
- Actualización asíncrona con permutación aleatoria en cada barrido.
