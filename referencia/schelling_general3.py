"""Modelo de segregación tipo Schelling con 3 poblaciones y matriz de aversión.

Tiene 3 tipos de agentes (1, 2 y 3) y una
matriz de aversión 3x3, donde `aversion[i][j]` es la aversión que un agente de
tipo i+1 siente por un vecino de tipo j+1. Por defecto no hay auto-aversión y
la aversión hacia cualquier otro tipo es 1:

    [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]

    Modelo_General3(n1, n2, n3, N, M, a11=0, a12=1, a13=1,
                     a21=1, a22=0, a23=1, a31=1, a32=1, a33=0, tolerancia=1.0)

La matriz se recibe como 9 coeficientes sueltos (a11..a33) en vez de una
matriz anidada porque la visualización de Solara reconstruye el modelo en
cada "Reset" llamando a `ModeloGeneral3(**model_params)`, y sus parámetros
deben coincidir uno a uno con las cajas de texto del panel "Model
Parameters" (ver `mesa.visualization.solara_viz._check_model_params`).

Un agente insatisfecho (insatisfacción > tolerancia) se mueve a la casilla
vacía más cercana donde su insatisfacción sería estrictamente menor que en su
posición actual; si no encuentra ninguna, se queda (misma regla que
schelling_general2.py y modelo.py).
"""

import math

import mesa

TIPOS = (1, 2, 3)

# Valores por defecto usados cuando n1/n2/n3/N/M llegan inválidos (texto no
# numérico, o negativos) desde la caja de texto de la visualización.
_DEFECTOS = {"n1": 500, "n2": 500, "n3": 500, "N": 40, "M": 40}


def _entero_valido(valor, nombre, minimo=0):
    """Convierte `valor` a int si es posible y >= minimo; si no, ignora el
    valor en silencio y devuelve el valor por defecto de `nombre`."""
    try:
        resultado = int(valor)
    except (TypeError, ValueError):
        return _DEFECTOS[nombre]
    return resultado if resultado >= minimo else _DEFECTOS[nombre]


def _flotante_valido(valor, defecto):
    """Convierte `valor` a float (aceptando texto con coma decimal, p.ej. de
    una caja de texto de la visualización); si no es un número válido,
    ignora el valor en silencio y devuelve `defecto`. Tolerancia y aversión
    son números reales cualesquiera (positivos, negativos o cero), así que
    aquí no se aplica ningún mínimo."""
    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return defecto


class AgenteGeneral3(mesa.Agent):
    def __init__(self, model, tipo: int, se_ha_movido: bool) -> None:
        super().__init__(model)
        self.tipo = tipo  # 1, 2 o 3
        self.se_ha_movido = se_ha_movido

    def insatisfaccion(self, pos, tipo):
        """Insatisfacción que un agente de `tipo` sentiría en `pos`: suma de
        las aversiones (según la matriz de aversión) hacia sus vecinos ahí."""
        vecinos = self.model.grid.get_neighbors(pos, moore=True, include_center=False)
        aversion = self.model.aversion
        return sum(aversion[tipo - 1][vecino.tipo - 1] for vecino in vecinos)

    def move(self):
        self.se_ha_movido = False
        possible_steps = []
        radio = 0
        max_iteraciones = max(self.model.grid.width, self.model.grid.height) // 2 + 1
        insatisfaccion_base = self.insatisfaccion(self.pos, self.tipo)

        if insatisfaccion_base > self.model.tolerancia:
            while not possible_steps and radio < max_iteraciones:
                radio += 1
                vecindario = self.model.grid.get_neighborhood(
                    self.pos, moore=True, include_center=False, radius=radio
                )
                empty_steps = [step for step in vecindario if self.model.grid.is_cell_empty(step)]
                possible_steps = [
                    step for step in empty_steps
                    if self.insatisfaccion(step, self.tipo) < insatisfaccion_base
                ]

        if possible_steps:
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
            self.se_ha_movido = True


class ModeloGeneral3(mesa.Model):
    def __init__(self, n1, n2, n3, N, M,
                 a11=0, a12=1, a13=0,
                 a21=0, a22=0, a23=1,
                 a31=1, a32=0, a33=0,
                 tolerancia=1.0, seed=None):
        super().__init__(seed=seed)
        n1 = _entero_valido(n1, "n1")
        n2 = _entero_valido(n2, "n2")
        n3 = _entero_valido(n3, "n3")
        N = _entero_valido(N, "N", minimo=1)
        M = _entero_valido(M, "M", minimo=1)
        self.aversion = [
            [_flotante_valido(a11, 0.0), _flotante_valido(a12, 1.0), _flotante_valido(a13, 1.0)],
            [_flotante_valido(a21, 1.0), _flotante_valido(a22, 0.0), _flotante_valido(a23, 1.0)],
            [_flotante_valido(a31, 1.0), _flotante_valido(a32, 1.0), _flotante_valido(a33, 0.0)],
        ]
        self.tolerancia = _flotante_valido(tolerancia, 1.0)
        self.num_agents = n1 + n2 + n3
        self.grid = mesa.space.SingleGrid(N, M, torus=True)
        self.running = True
        self.steps_to_equilibrium = 0
        self.in_equilibrium = False

        # Datos guardados en cada paso (ver calcular_metricas)
        self.insatisfaccion_tipo1 = 0.0
        self.insatisfaccion_tipo2 = 0.0
        self.insatisfaccion_tipo3 = 0.0
        self.prop_11 = 0.0
        self.prop_22 = 0.0
        self.prop_33 = 0.0
        self.prop_12 = 0.0
        self.prop_13 = 0.0
        self.prop_23 = 0.0
        self.entropia = 0.0

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Insatisfaccion_Tipo1": "insatisfaccion_tipo1",
                "Insatisfaccion_Tipo2": "insatisfaccion_tipo2",
                "Insatisfaccion_Tipo3": "insatisfaccion_tipo3",
                "Prop_11": "prop_11",
                "Prop_22": "prop_22",
                "Prop_33": "prop_33",
                "Prop_12": "prop_12",
                "Prop_13": "prop_13",
                "Prop_23": "prop_23",
                "Entropia": "entropia",
            }
        )

        for _ in range(n1): AgenteGeneral3(model=self, tipo=1, se_ha_movido=False)
        for _ in range(n2): AgenteGeneral3(model=self, tipo=2, se_ha_movido=False)
        for _ in range(n3): AgenteGeneral3(model=self, tipo=3, se_ha_movido=False)

        all_positions = [(x, y) for x in range(N) for y in range(M)]
        self.random.shuffle(all_positions)
        for agent, pos in zip(self.agents, all_positions):
            self.grid.place_agent(agent, pos)

        self.calcular_metricas()

    def calcular_metricas(self):
        """Calcula y guarda: insatisfacción promedio por tipo, proporción de
        interacciones por cada par de tipos, y entropía local promedio (ver
        docstring del módulo para la definición de cada una)."""
        suma = {t: 0.0 for t in TIPOS}
        conteo = {t: 0 for t in TIPOS}
        for agent in self.agents:
            suma[agent.tipo] += agent.insatisfaccion(agent.pos, agent.tipo)
            conteo[agent.tipo] += 1
        self.insatisfaccion_tipo1 = suma[1] / conteo[1] if conteo[1] > 0 else 0.0
        self.insatisfaccion_tipo2 = suma[2] / conteo[2] if conteo[2] > 0 else 0.0
        self.insatisfaccion_tipo3 = suma[3] / conteo[3] if conteo[3] > 0 else 0.0

        # Proporciones de interacciones (mismo patrón que modelo.py: cada par
        # no ordenado se cuenta dos veces al recorrer por ambos extremos, así
        # que las entradas de la diagonal se dividen entre 2).
        contact_counts = {a: {b: 0 for b in TIPOS} for a in TIPOS}
        total_contactos = 0
        entropias = []
        for agent in self.agents:
            vecinos = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            for vecino in vecinos:
                contact_counts[agent.tipo][vecino.tipo] += 1
                total_contactos += 1

            if not vecinos:
                entropias.append(0.0)
                continue
            conteo_local = {t: 0 for t in TIPOS}
            for vecino in vecinos:
                conteo_local[vecino.tipo] += 1
            total_local = len(vecinos)
            h = 0.0
            for t in TIPOS:
                p = conteo_local[t] / total_local
                if p > 0:
                    h -= p * math.log2(p)
            entropias.append(h)

        total_contactos /= 2
        for t in TIPOS:
            contact_counts[t][t] /= 2

        if total_contactos > 0:
            self.prop_11 = contact_counts[1][1] / total_contactos
            self.prop_22 = contact_counts[2][2] / total_contactos
            self.prop_33 = contact_counts[3][3] / total_contactos
            self.prop_12 = contact_counts[1][2] / total_contactos
            self.prop_13 = contact_counts[1][3] / total_contactos
            self.prop_23 = contact_counts[2][3] / total_contactos
        else:
            self.prop_11 = self.prop_22 = self.prop_33 = 0.0
            self.prop_12 = self.prop_13 = self.prop_23 = 0.0

        self.entropia = sum(entropias) / len(entropias) if entropias else 0.0

    def check_equilibrium(self) -> bool:
        return not any(agent.se_ha_movido for agent in self.agents)

    def step(self):
        if not self.in_equilibrium:
            self.agents.shuffle_do("move")
            self.calcular_metricas()

            if self.check_equilibrium():
                self.in_equilibrium = True
                self.running = False
            else:
                self.steps_to_equilibrium += 1

            self.datacollector.collect(self)
