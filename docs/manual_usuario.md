# Manual de usuario — `schelling_matricial` 0.1

Este manual explica cómo usar el paquete para simular la extensión matricial
del modelo de Schelling. La definición formal del modelo está en
[`especificacion_fase0.md`](especificacion_fase0.md) y el marco teórico en
`programa_investigacion_schelling_matricial.tex`. Aquí nos centramos en el uso.

## Índice

1. [Instalación](#1-instalación)
2. [Primer ejemplo](#2-primer-ejemplo)
3. [El modelo en una página](#3-el-modelo-en-una-página)
4. [Configuración: `Config`](#4-configuración-config)
5. [Ejecutar una simulación: `Simulacion`](#5-ejecutar-una-simulación-simulacion)
6. [El resultado](#6-el-resultado)
7. [Réplicas y ficheros CSV](#7-réplicas-y-ficheros-csv)
8. [Herramientas de inspección](#8-herramientas-de-inspección)
9. [Recetas](#9-recetas)
10. [Errores frecuentes](#10-errores-frecuentes)
11. [Limitaciones de la versión 0.1](#11-limitaciones-de-la-versión-01)

---

## 1. Instalación

Requisitos: Python ≥ 3.10. La única dependencia es `numpy`. `pytest` solo hace
falta para las pruebas.

```bash
cd schelling_matricial
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

La opción `-e` (modo editable) hace que los cambios en `src/` se vean al
momento, sin reinstalar. Para comprobar que todo funciona:

```bash
pytest
```

Todas las pruebas deben pasar.

---

## 2. Primer ejemplo

El modelo clásico de Schelling tiene dos tipos. Un agente está incómodo si más
de la mitad de sus vecinos ocupados son del otro tipo.

```python
from schelling_matricial import Config, Simulacion

cfg = Config(
    L=40,                    # retícula 40×40 (1600 casillas)
    poblaciones=(700, 700),  # 700 agentes de cada tipo; 200 casillas vacías
    H=[[0, 1],
       [1, 0]],              # un vecino distinto molesta 1; uno igual, 0
    tau=0.5,                 # umbral: incómodo si la incomodidad > 0.5
    modelo="P",              # modelo promediado (proporciones)
    seed=123,                # semilla: misma semilla, mismo resultado
)

sim = Simulacion(cfg)
resultado = sim.ejecutar()

print(resultado["motivo_parada"])    # 'punto_fijo'
print(resultado["tipo_punto_fijo"])  # 'satisfecho'
print(resultado["barridos"])         # barridos realizados
print(resultado["s"])                # índice de segregación
```

La retícula final queda en `sim.reticula`, un array `L×L` en el que `0` es una
casilla vacía y `1..n` es el tipo del agente que la ocupa. Para verla:

```python
import matplotlib.pyplot as plt   # no es dependencia del paquete
plt.imshow(sim.reticula, cmap="viridis", interpolation="nearest")
plt.show()
```

---

## 3. El modelo en una página

**Espacio.** Es una retícula toroidal `L×L`: los bordes se tocan, así que no
hay casillas de borde. Cada agente tiene `k` vecinos:

| Vecindad | Radio `r` | `k` |
|---|---|---|
| Moore (`"moore"`) | 1, 2, 3 | 8, 24, 48 |
| von Neumann (`"von_neumann"`) | 1, 2, 3 | 4, 12, 24 |

Si una casilla `x` tiene `n_j(x)` vecinos de tipo `j`, sus vecinos ocupados son
`n(x) = Σ_j n_j(x)` y sus vecinos vacíos, `n_vac(x) = k − n(x)`.

**Matriz `H`.** `H[i][j]` es cuánto molesta a un agente del tipo `i+1` tener un
vecino del tipo `j+1`. Un valor positivo significa rechazo, uno negativo,
atracción y cero, indiferencia. Una columna adicional (la última) indica cuánto
le molesta una casilla vecina vacía.

**Incomodidad** de un agente de tipo `i` en la casilla `x`:

- **Modelo P (promediado):** la media de `H[i][j]` sobre los vecinos ocupados,

  `I_P = (Σ_j n_j·H[i][j]) / n(x)`.

  Si el agente no tiene vecinos ocupados (está *aislado*), `I_P = iota[i]`.
  Los vacíos no cuentan.
- **Modelo S (de suma):** la suma sobre toda la vecindad, vacíos incluidos,

  `I_S = Σ_j n_j·H[i][j] + n_vac·H[i][vac]`.

  Es el modelo del código antiguo del repositorio `mesa-python`.

**Dinámica.** Cada *barrido* recorre a todos los agentes en un orden aleatorio
nuevo. Cada agente actúa sobre el estado ya actualizado por los anteriores:

1. Si su incomodidad es ≤ `tau[i]`, está **cómodo** y no hace nada.
2. Si no, busca las casillas vacías donde estaría **estrictamente mejor**. Para
   evaluar una casilla no se cuenta a sí mismo como vecino, aunque esa casilla
   esté junto a su posición actual.
3. Entre ellas elige las más cercanas (distancia de Chebyshev en el toro) y,
   de estas, una al azar. Si no hay ninguna que mejore, se queda donde está.

**Parada.** Tras cada barrido se comprueba, en este orden:

| `motivo_parada` | Condición |
|---|---|
| `"punto_fijo"` | Nadie se movió en el barrido. `tipo_punto_fijo` vale `"satisfecho"` si todos están cómodos y `"bloqueado"` si queda alguno incómodo sin mejora posible. |
| `"estado_repetido"` | La retícula ya había aparecido antes (el estado inicial incluido). Indica un posible ciclo. Solo para si `parar_en_repeticion=True`. |
| `"t_max"` | Se alcanzaron `T_max` barridos. |

---

## 4. Configuración: `Config`

Todos los parámetros de una simulación se agrupan en un `Config`. Al crearlo
se validan y, si algo no cuadra, se lanza un `ValueError` con un mensaje
explicativo.

| Parámetro | Tipo | Por defecto | Significado |
|---|---|---|---|
| `L` | `int` | — | Lado de la retícula. Debe cumplir `L > 2·radio`. |
| `poblaciones` | tupla de `int` | — | `(N_1, …, N_n)`: agentes de cada tipo. Su longitud define `n`. La suma debe ser ≤ `L²`. |
| `H` | matriz | — | `n×n` o `n×(n+1)`. Con `n×n` se añade una columna de vacío llena de ceros. |
| `tau` | número o vector | `0.5` | Umbral de comodidad: común o uno por tipo. |
| `iota` | número o vector | `0` | Incomodidad de un agente aislado (solo en el modelo P). |
| `modelo` | `"P"` o `"S"` | `"P"` | Modelo promediado o de suma. |
| `vecindad` | `"moore"` o `"von_neumann"` | `"moore"` | Forma de la vecindad. |
| `radio` | `int` | `1` | Radio de la vecindad. |
| `busqueda` | `str` | `"cercano"` | Regla para elegir destino. En v0.1 solo existe `"cercano"`. |
| `T_max` | `int` | `1000` | Máximo de barridos. |
| `seed` | `int` o `None` | `None` | Semilla. Con `None` cada ejecución es distinta. |
| `parar_en_repeticion` | `bool` | `True` | Parar al detectar un estado repetido. |
| `cada` | `int` o `None` | `None` | Si se indica, guarda los observables cada `cada` barridos (ver §6.3). |

### 4.1 Cómo escribir `H`

Las filas son el tipo que *siente* la incomodidad y las columnas, el vecino que
la *provoca*. Con tres tipos y columna de vacío:

```python
#        vecino: tipo1 tipo2 tipo3 vacío
H = [
    [ 0,    1,    1,    0 ],   # tipo 1: rechaza a los tipos 2 y 3
    [-1,    0,    0,    0 ],   # tipo 2: le atrae el tipo 1
    [ 1,    1,    0,  0.5 ],   # tipo 3: rechaza a 1 y 2, y un poco a los vacíos
]
```

Los números pueden ser enteros, decimales o `fractions.Fraction`. La columna
de vacío solo tiene efecto en el modelo S. En el modelo P se ignora.

### 4.2 Aritmética exacta

Los agentes nunca deciden comparando decimales en coma flotante. Al crear el
`Config`, `H`, `tau` e `iota` se convierten en fracciones exactas (`0.1` pasa a
ser `1/10`) y luego en enteros, multiplicándolos por un factor común,
`cfg.escala`. Multiplicar todos los parámetros por el mismo número positivo no
cambia la dinámica.

La consecuencia práctica es que un agente que está **exactamente** en el
umbral cuenta como cómodo. Por ejemplo, con
`H=[[0.4, 0.6], [0.6, 0.4]]` y `tau=0.5`, un agente con 4 vecinos iguales y 4
distintos tiene incomodidad `0.5` exacta.

Los decimales se leen con precisión de 10⁻⁶. Para valores como 1/3, lo más
seguro es pasar `Fraction(1, 3)`.

---

## 5. Ejecutar una simulación: `Simulacion`

```python
sim = Simulacion(cfg)          # coloca los agentes al azar (según la semilla)
resultado = sim.ejecutar()     # barre hasta una condición de parada
```

### 5.1 Barrido a barrido

Para seguir la evolución paso a paso se puede llamar a `barrido()`, que hace
un solo barrido y devuelve el número de movimientos:

```python
sim = Simulacion(cfg)
estados = [sim.reticula.copy()]
while True:
    movimientos = sim.barrido()
    estados.append(sim.reticula.copy())   # copy(): la retícula cambia en su sitio
    if movimientos == 0:
        break
print(len(estados), "estados;", sim.movimientos_totales, "movimientos")
```

`ejecutar()` puede llamarse después de unos barridos manuales. Continúa desde
el estado actual, y `T_max` cuenta el total de barridos.

### 5.2 Empezar desde una retícula concreta

En lugar de colocar los agentes al azar, se puede pasar una retícula inicial.
Sus poblaciones deben coincidir con `cfg.poblaciones`:

```python
import numpy as np

ret = np.zeros((7, 7), dtype=int)
ret[3, 3] = 1
ret[3, 4] = 2
cfg = Config(L=7, poblaciones=(1, 1), H=[[0, 1], [1, 0]], tau=0.5, seed=0)
sim = Simulacion(cfg, reticula=ret)
```

La semilla sigue controlando el orden de los barridos y los desempates.

### 5.3 Observar cada movimiento

`barrido()` y `ejecutar()` aceptan una función `al_mover` que se llama tras
cada movimiento con estos argumentos:

```
al_mover(agente, tipo, origen, destino, I_origen, I_destino)
```

- `agente`: índice del agente (posición en `sim.posiciones`).
- `tipo`: su tipo (`1..n`).
- `origen`, `destino`: casillas `(fila, columna)`.
- `I_origen`, `I_destino`: incomodidades exactas (`Fraction`, en la escala
  original de `H`) antes y después del movimiento.

```python
registro = []
sim.ejecutar(al_mover=lambda a, t, x, y, Ix, Iy: registro.append((t, x, y, Ix - Iy)))
```

---

## 6. El resultado

`ejecutar()` devuelve un diccionario con tres grupos de claves.

### 6.1 Parámetros y metadatos

Contiene todos los parámetros de `Config` (`L`, `poblaciones`, `H` ya extendida
a `n×(n+1)`, `tau` e `iota` como vectores, `modelo`, `vecindad`, `radio`, `k`,
`busqueda`, `T_max`, `seed`, `parar_en_repeticion`), además de:

| Clave | Significado |
|---|---|
| `version` | Versión del paquete. |
| `git` | Hash del commit de git, o `None` si no se puede obtener. |
| `barridos` | Barridos realizados, incluido el último. En un punto fijo, ese último barrido es el que ya no tiene movimientos, así que un estado que ya era estable da `barridos = 1`. |
| `movimientos_totales` | Número total de cambios de casilla. |
| `motivo_parada` | `"punto_fijo"`, `"estado_repetido"` o `"t_max"`. |
| `tipo_punto_fijo` | `"satisfecho"`, `"bloqueado"` o `None` si no se paró en un punto fijo. |
| `repeticion_detectada` | `True` si en algún momento se repitió un estado, aunque no se parase por ello. |
| `barrido_repeticion` | Primer barrido en que se detectó una repetición, o `None`. |

### 6.2 Observables del estado final

En los arrays, el índice `0` corresponde al tipo 1, el índice `1` al tipo 2 y
así sucesivamente.

| Clave | Forma | Significado |
|---|---|---|
| `u` | `(n,)` | Fracción de agentes incómodos de cada tipo. |
| `I_media` | `(n,)` | Incomodidad media de cada tipo, en la escala original de `H`. |
| `C_occ` | `(n, n)` | `C_occ[i, j]`: media, sobre los agentes de tipo `i`, de la fracción de sus vecinos ocupados que son de tipo `j`. Los aislados aportan 0. |
| `C_full` | `(n, n+1)` | Igual, pero dividiendo por `k`. La última columna es la fracción de vecinos vacíos. |
| `aislados` | `(n,)` | Agentes sin ningún vecino ocupado, por tipo. |
| `frac_aislados` | escalar | Fracción total de agentes aislados. |
| `h` | escalar | Homofilia: `Σ_i (N_i/N)·C_occ[i, i]`. |
| `rho_int` | escalar | Fracción de pares de vecinos ocupados que son de tipos distintos. |
| `rho_int_azar` | escalar | Valor esperado de `rho_int` con mezcla aleatoria: `1 − Σ_i (N_i/N)²`. |
| `s` | escalar | Índice de segregación: `1 − rho_int / rho_int_azar`. Un valor positivo indica segregación, uno cercano a 0, mezcla aleatoria, y uno negativo, mezcla alternada. |

Si un tipo tiene población 0, sus observables valen `nan`.

### 6.3 Serie temporal

Con `cada=m` en el `Config`, el resultado incluye la clave `serie`: una lista
de diccionarios con los observables en los barridos `0, m, 2m, …`. Cada uno
lleva además `barrido` y `movimientos_totales`.

```python
cfg = Config(L=40, poblaciones=(700, 700), H=[[0, 1], [1, 0]], tau=0.3, seed=1, cada=1)
r = Simulacion(cfg).ejecutar()
barridos = [e["barrido"] for e in r["serie"]]
segregacion = [e["s"] for e in r["serie"]]
```

La serie no se guarda en el CSV.

---

## 7. Réplicas y ficheros CSV

`ejecutar_replicas` repite la simulación con varias semillas. El resto de
parámetros se toman del `Config`, cuya `seed` se ignora:

```python
from schelling_matricial import Config, ejecutar_replicas

cfg = Config(L=40, poblaciones=(700, 700), H=[[0, 1], [1, 0]], tau=0.5)
filas = ejecutar_replicas(cfg, semillas=range(50), salida="resultados/clasico_tau05.csv")
```

- Devuelve una lista de diccionarios «planos», uno por réplica.
- Con `salida=`, escribe además un CSV con una fila por réplica y crea la
  carpeta si no existe. La carpeta `resultados/` está en `.gitignore`.

**Nombres de columna.** Las matrices y los vectores se aplanan con índices que
empiezan en 1, y el vacío se llama `vac`:

| Dato | Columnas |
|---|---|
| `u` | `u_1`, `u_2`, … |
| `C_occ` | `C_occ_1_1`, `C_occ_1_2`, … |
| `C_full` | `C_full_1_1`, …, `C_full_1_vac`, … |
| `H` | `H_1_1`, …, `H_1_vac`, … |
| `poblaciones`, `tau`, `iota` | `poblaciones_1`, `tau_1`, `iota_1`, … |

**Análisis con pandas** (opcional, no es dependencia del paquete):

```python
import pandas as pd
df = pd.DataFrame(filas)                      # o pd.read_csv("resultados/...")
print(df["s"].mean(), df["s"].std())
print(df["motivo_parada"].value_counts())
```

Para aplanar un resultado suelto se usa `aplanar(resultado)`.

### 7.1 Un barrido de parámetros sencillo

```python
import dataclasses
filas = []
for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
    cfg_tau = dataclasses.replace(cfg, tau=tau)
    filas += ejecutar_replicas(cfg_tau, semillas=range(20))
```

`dataclasses.replace` crea una copia del `Config` con los campos cambiados y
vuelve a validarla.

---

## 8. Herramientas de inspección

Métodos y atributos de `Simulacion`:

| Uso | Devuelve |
|---|---|
| `sim.reticula` | Array `L×L` con el estado actual. |
| `sim.posiciones` | Array `(N, 2)` con la `(fila, columna)` de cada agente. |
| `sim.tipos` | Array `(N,)` con el tipo (`1..n`) de cada agente. |
| `sim.incomodidades()` | Lista con la incomodidad exacta (`Fraction`) de cada agente. |
| `sim.comodos()` | Array booleano: qué agentes están cómodos. |
| `sim.incomodidad(pos)` | Incomodidad del agente que ocupa `pos`. |
| `sim.incomodidad(pos, tipo=t, excluir=x)` | Incomodidad que tendría un agente de tipo `t` en `pos` si la casilla `x` estuviera vacía. Sirve para evaluar a mano el destino de un agente que está en `x`. |
| `sim.barridos`, `sim.movimientos_totales` | Contadores acumulados. |

Funciones del paquete:

| Uso | Devuelve |
|---|---|
| `calcular_observables(reticula, cfg)` | Los observables del §6.2 para cualquier retícula. |
| `phi(reticula, S, vecindad="moore", radio=1)` | Potencial de aristas `Φ = Σ S[t(u), t(v)]` sobre los pares de vecinos ocupados. `S` debe ser simétrica. Con `S` entera, el resultado es un entero exacto. |

---

## 9. Recetas

### 9.1 Modelo clásico: segregación frente a tolerancia

```python
import numpy as np
from schelling_matricial import Config, ejecutar_replicas

L = 40
N = int(0.9 * L * L)                          # 10 % de vacíos
for tau in (0.3, 0.5, 0.7):
    cfg = Config(L=L, poblaciones=(N // 2, N - N // 2), H=[[0, 1], [1, 0]], tau=tau)
    s = [f["s"] for f in ejecutar_replicas(cfg, semillas=range(10))]
    print(f"tau={tau}: s = {np.mean(s):.2f} ± {np.std(s):.2f}")
```

Cuanto menor es `tau` (más intolerancia), mayor es la segregación `s`.

### 9.2 Modelo S con población neutral

Tres tipos. Los extremos (tipos 1 y 3) se rechazan entre sí y el neutral
(tipo 2) rechaza a ambos:

```python
H_neu = [[0, 0, 1],
         [1, 0, 1],
         [1, 0, 0]]
cfg = Config(L=30, poblaciones=(250, 250, 250), H=H_neu, tau=2, modelo="S", seed=0)
r = Simulacion(cfg).ejecutar()
print(r["motivo_parada"], r["barridos"], r["h"])
```

En el modelo S, `tau` se mide en las unidades de la suma. Con
`H[i][j] ∈ {0, 1}`, `tau=2` significa «tolera hasta 2 vecinos molestos».

### 9.3 Umbrales distintos por tipo

```python
cfg = Config(L=40, poblaciones=(700, 700), H=[[0, 1], [1, 0]], tau=[0.3, 0.7], seed=0)
r = Simulacion(cfg).ejecutar()
print(r["u"], r["C_occ"])
```

### 9.4 Comprobar el potencial en el modelo S

Si `H[i][j] − H[i][vac] = w_i · S[i][j]`, con `S` simétrica y pesos `w_i > 0`,
cada movimiento cambia `Φ` exactamente en `(I_destino − I_origen) / w_i`, y `Φ`
decrece estrictamente:

```python
from schelling_matricial import Config, Simulacion, phi

S, w = [[0, 1], [1, 0]], (1, 3)
H = [[0, 1, 0],
     [3, 0, 0]]                 # H[i][j] = w_i·S[i][j], H[i][vac] = 0
cfg = Config(L=20, poblaciones=(150, 150), H=H, tau=(2, 5), modelo="S", seed=0)
sim = Simulacion(cfg)
valores = [phi(sim.reticula, S)]
sim.ejecutar(al_mover=lambda *args: valores.append(phi(sim.reticula, S)))
assert all(b < a for a, b in zip(valores, valores[1:]))
```

Recalcular `phi` en cada movimiento es lento: usa esta receta solo en
retículas pequeñas.

### 9.5 Buscar ciclos

Con `parar_en_repeticion=False`, la simulación sigue hasta `T_max`, pero sigue
anotando si hubo repeticiones:

```python
cfg = Config(L=5, poblaciones=(1, 1), H=[[0, -1], [1, 0]], tau=(-1, 0), modelo="S",
             seed=0, T_max=500, parar_en_repeticion=False)
r = Simulacion(cfg).ejecutar()
print(r["motivo_parada"], r["repeticion_detectada"], r["barrido_repeticion"])
```

Aquí el tipo 1 persigue al 2, que huye de él. Como la dinámica es aleatoria,
un estado repetido es un **indicio** de ciclo, no una prueba.

---

## 10. Errores frecuentes

| Mensaje o síntoma | Causa y solución |
|---|---|
| `ValueError: se necesita L > 2r` | La retícula es demasiado pequeña para el radio: un vecino aparecería dos veces. Aumenta `L` o reduce `radio`. |
| `ValueError: H debe tener n filas` / `cada fila de H debe tener n o n+1 columnas` | El número de filas de `H` debe coincidir con el de `poblaciones`. |
| `ValueError: la suma de poblaciones supera L^2` | Hay más agentes que casillas. |
| `ValueError: tau debe ser escalar o tener longitud n` | El vector de umbrales tiene una longitud distinta del número de tipos. |
| `NotImplementedError: la regla de búsqueda 'mejor' aún no está implementada` | En v0.1 solo existe `busqueda="cercano"`. |
| `OverflowError: coeficientes demasiado grandes` | Los parámetros tienen denominadores muy distintos y el factor común es enorme. Simplifica los valores (por ejemplo, usa fracciones con denominadores pequeños). |
| Dos ejecuciones dan resultados distintos | Falta la `seed` en el `Config`. |
| Una lista de retículas guardadas sale toda igual | `sim.reticula` se modifica en su sitio: guarda `sim.reticula.copy()`. |
| En el modelo P nadie se mueve aunque haya incómodos | Puede ser un punto fijo **bloqueado** (no hay casillas que mejoren). Compruébalo con `tipo_punto_fijo`. |
| En el modelo P, la columna de vacío de `H` no hace nada | Es lo esperado: en el modelo P los vacíos no cuentan. Para agentes aislados se usa `iota`. |

---

## 11. Limitaciones de la versión 0.1

- Solo retícula toroidal cuadrada; todavía no hay redes (Erdős–Rényi).
- Solo existe la regla de búsqueda `"cercano"`. Las reglas `"aleatorio"` y
  `"mejor"` están previstas.
- No hay umbrales individuales heterogéneos (solo por tipo).
- No hay visualización integrada ni herramientas de barrido masivo de
  parámetros.
- Rendimiento orientativo: un barrido con `L = 80`, un 10 % de vacíos y Moore
  `r = 1` tarda menos de medio segundo en un portátil. Los primeros barridos
  son los más lentos, porque es cuando más agentes se mueven.
