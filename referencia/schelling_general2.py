"""Modelo de segregación tipo Schelling con 2 poblaciones y matriz de aversión.

Generaliza modelo.py: en vez de una tolerancia fija por vecino "incómodo", cada
tipo de agente siente una aversión (posiblemente distinta) hacia cada tipo,
dada por una matriz 2x2 `aversion`, donde `aversion[i][j]` es la aversión que
un agente de tipo i+1 siente por un vecino de tipo j+1. La insatisfacción de un
agente es la suma de esas aversiones sobre todos sus vecinos actuales.

    Modelo_General2(n1, n2, N, M, a11=0, a12=1, a21=1, a22=0, tolerancia=1.0)

La matriz se recibe como 4 coeficientes sueltos (a11..a22) en vez de una
matriz anidada porque la visualización de Solara reconstruye el modelo en
cada "Reset" llamando a `ModeloGeneral2(**model_params)`, y sus parámetros
deben coincidir uno a uno con las cajas de texto del panel "Model
Parameters" (ver `mesa.visualization.solara_viz._check_model_params`).

Un agente insatisfecho (insatisfacción > tolerancia) se mueve a la casilla
vacía más cercana donde su insatisfacción sería estrictamente menor que en su
posición actual (igual que modelo.py); si no encuentra ninguna, se queda.
"""

import math

import mesa


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


def _entero_valido(valor, defecto):
    """Como `_flotante_valido`, pero para n1, n2, N y M (recibidos como texto
    desde las cajas de la visualización); si no es un entero válido, ignora
    el valor en silencio y devuelve `defecto`."""
    try:
        return int(float(str(valor).replace(",", ".")))
    except (TypeError, ValueError):
        return defecto


class AgenteGeneral2(mesa.Agent):
    def __init__(self, model, tipo: int, se_ha_movido: bool) -> None:
        super().__init__(model)
        self.tipo = tipo  # 1 o 2
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


class ModeloGeneral2(mesa.Model):
    def __init__(self, n1, n2, N, M, a11=0, a12=1, a21=1, a22=0, tolerancia=1.0, seed=None):
        super().__init__(seed=seed)
        n1 = _entero_valido(n1, 500)
        n2 = _entero_valido(n2, 500)
        N = _entero_valido(N, 40)
        M = _entero_valido(M, 40)
        self.aversion = [
            [_flotante_valido(a11, 0.0), _flotante_valido(a12, 1.0)],
            [_flotante_valido(a21, 1.0), _flotante_valido(a22, 0.0)],
        ]
        self.tolerancia = _flotante_valido(tolerancia, 1.0)
        self.num_agents = n1 + n2
        self.grid = mesa.space.SingleGrid(N, M, torus=True)
        self.running = True
        self.steps_to_equilibrium = 0
        self.in_equilibrium = False

        # Datos guardados en cada paso (ver calcular_metricas)
        self.insatisfaccion_tipo1 = 0.0
        self.insatisfaccion_tipo2 = 0.0
        self.prop_11 = 0.0
        self.prop_22 = 0.0
        self.prop_12 = 0.0
        self.entropia = 0.0

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Insatisfaccion_Tipo1": "insatisfaccion_tipo1",
                "Insatisfaccion_Tipo2": "insatisfaccion_tipo2",
                "Prop_11": "prop_11",
                "Prop_22": "prop_22",
                "Prop_12": "prop_12",
                "Entropia": "entropia",
            }
        )

        for _ in range(n1): AgenteGeneral2(model=self, tipo=1, se_ha_movido=False)
        for _ in range(n2): AgenteGeneral2(model=self, tipo=2, se_ha_movido=False)

        all_positions = [(x, y) for x in range(N) for y in range(M)]
        self.random.shuffle(all_positions)
        for agent, pos in zip(self.agents, all_positions):
            self.grid.place_agent(agent, pos)

        self.calcular_metricas()

    def calcular_metricas(self):
        """Calcula y guarda: insatisfacción promedio por tipo, proporción de
        interacciones 1-1/2-2/1-2, y entropía local promedio (ver docstring
        del módulo para la definición de cada una)."""
        suma = {1: 0.0, 2: 0.0}
        conteo = {1: 0, 2: 0}
        for agent in self.agents:
            suma[agent.tipo] += agent.insatisfaccion(agent.pos, agent.tipo)
            conteo[agent.tipo] += 1
        self.insatisfaccion_tipo1 = suma[1] / conteo[1] if conteo[1] > 0 else 0.0
        self.insatisfaccion_tipo2 = suma[2] / conteo[2] if conteo[2] > 0 else 0.0

        # Proporciones de interacciones (mismo patrón que modelo.py: cada par
        # no ordenado se cuenta dos veces al recorrer por ambos extremos, así
        # que las entradas de la diagonal se dividen entre 2).
        contact_counts = {1: {1: 0, 2: 0}, 2: {1: 0, 2: 0}}
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
            conteo_local = {1: 0, 2: 0}
            for vecino in vecinos:
                conteo_local[vecino.tipo] += 1
            total_local = len(vecinos)
            h = 0.0
            for t in (1, 2):
                p = conteo_local[t] / total_local
                if p > 0:
                    h -= p * math.log2(p)
            entropias.append(h)

        total_contactos /= 2
        contact_counts[1][1] /= 2
        contact_counts[2][2] /= 2

        if total_contactos > 0:
            self.prop_11 = contact_counts[1][1] / total_contactos
            self.prop_22 = contact_counts[2][2] / total_contactos
            self.prop_12 = contact_counts[1][2] / total_contactos
        else:
            self.prop_11 = self.prop_22 = self.prop_12 = 0.0

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
