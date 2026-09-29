import mesa


# ========== AGENTE ==========
class Persona(mesa.Agent):
    def __init__(self, model, tipo: int, se_ha_movido: bool) -> None:
        super().__init__(model)
        self.tipo = tipo  # tipos (-1), (+1), hostiles y 0 neutrales
        self.se_ha_movido = se_ha_movido # para actualizar TRUE cuando el agente cambia de posición


# La incomodidad es de un agente (self) en una casilla (pos) asumiendo que el agente es de un tipo. incomodidad(agente,posición,tipo) es el número
# de vecinos incómodos que tendría el agente si se encontrara en la posición.
    
    def incomodidad(self, pos, tipo):
        contador = 0
        vecindad = self.model.grid.get_neighbors(pos, moore=True, include_center=False)
        if tipo == 1:
            for vecino in vecindad:
                if vecino.tipo == -1: contador += 1
        elif tipo == -1:
            for vecino in vecindad:
                if vecino.tipo == 1: contador += 1
        elif tipo == 0:
            for vecino in vecindad: 
                contador += abs(vecino.tipo)
        return contador

# Esta es la función fundamental que recoloca a un agente, siempre que su incomodidad esté por encima de su tolerancia. 
# Para ello busca la casilla vacía más cercana en la que tendría una incomodidad menor a la actual, y se mueve ahí.
# Si no encuentra ninguna casilla con mejor situación, se resigna y se queda donde está.
    
    def move(self):
        self.se_ha_movido = False
        possible_steps = []
        radio = 0
        max_iteraciones = self.model.grid.width // 2 + 1
        incomodidad_base = self.incomodidad(self.pos, self.tipo)

        if incomodidad_base > self.model.tolerance: 
            while not possible_steps and radio < max_iteraciones:
                radio += 1
                vecindario = self.model.grid.get_neighborhood(
                    self.pos, moore=True, include_center=False, radius=radio
                )
                empty_steps = [step for step in vecindario if self.model.grid.is_cell_empty(step)]
                possible_steps = [step for step in empty_steps if self.incomodidad(step, self.tipo) < incomodidad_base]

        if possible_steps:
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
            self.se_ha_movido = True

# ========== MODELO ==========

class Modelo_Desplazamiento(mesa.Model):
    def __init__(self, N1, N2, N3, width, height, tol, seed=None):
        super().__init__(seed=seed)
        self.num_agents = N1 + N2 + N3
        self.grid = mesa.space.SingleGrid(width, height, True)
        self.running = True
        self.tolerance = tol
        self.steps_to_equilibrium = 0
        self.in_equilibrium = False
        self.contact_measure = {}
        self.same_type_contact_ratio = 0.0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Steps_to_Equilibrium": "steps_to_equilibrium",
                "Contact_Measure": "contact_measure",
                "In_Equilibrium": "in_equilibrium",
                "Same_Type_Contact_Ratio": "same_type_contact_ratio",
            }
        )

        for i in range(N1): Persona(model=self, tipo=1, se_ha_movido=False)
        for i in range(N2): Persona(model=self, tipo=0, se_ha_movido=False)
        for i in range(N3): Persona(model=self, tipo=-1, se_ha_movido=False)


        # Obtener todas las posiciones del grid
        all_positions = [
            (x, y)
            for x in range(self.grid.width)
            for y in range(self.grid.height)
        ]

        # Mezclar aleatoriamente
        self.random.shuffle(all_positions)

        # Asignar una posición por agente
        for agent, pos in zip(self.agents, all_positions):
            self.grid.place_agent(agent, pos)
        
        self.calculate_contact_measure()

    def calculate_contact_measure(self):
        """Calcula la medida de contacto desagregada por tipo de agente"""
        contact_counts = {
            -1: {-1: 0, 0: 0, 1: 0},
             0: {-1: 0, 0: 0, 1: 0},
             1: {-1: 0, 0: 0, 1: 0},
        }
        Total_contacts = 0

        for agent in self.agents:
            neighbors = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                contact_counts[agent.tipo][neighbor.tipo] += 1
                Total_contacts += 1

        Total_contacts = Total_contacts / 2

        for i in [-1,0,1]: contact_counts[i][i] = contact_counts[i][i] / 2
        
        # Guardamos proporciones
        self.contact_measure = {}
        for agent_type in [-1, 0, 1]:
            self.contact_measure[agent_type] = {}
            for neighbor_type in [-1, 0, 1]:
                if Total_contacts > 0:
                    self.contact_measure[agent_type][neighbor_type] = (
                        contact_counts[agent_type][neighbor_type] / Total_contacts
                    )
                else:
                    self.contact_measure[agent_type][neighbor_type] = 0

        # Proporción de todos los contactos que ocurren entre agentes del mismo tipo
        # (indicador escalar de segregación, útil para graficar su evolución en el tiempo)
        self.same_type_contact_ratio = sum(
            self.contact_measure[t][t] for t in [-1, 0, 1]
        )

    def check_equilibrium(self) -> bool:
        """Verifica si el modelo ha alcanzado el equilibrio"""
        for agent in self.agents:
            if agent.se_ha_movido:
                return False
        return True
    
    def step(self):
        if not self.in_equilibrium:
            self.agents.shuffle_do("move")
            self.calculate_contact_measure()
            
            if self.check_equilibrium():
                self.in_equilibrium = True
                self.running = False
            else:
                self.steps_to_equilibrium += 1
            
            self.datacollector.collect(self)
