
import numpy as np
import math

class BaseHeatDiffusion:
    """
    Classe base para a simulação de difusão de calor 2D usando o método de diferenças finitas.
    Contém a lógica comum para inicialização, aplicação de condições de contorno e o cálculo
    da atualização da temperatura de uma célula.

    Esta classe implementa a equação do calor 2D (transiente) na forma explícita:
    dT/dt = alpha * (d^2T/dx^2 + d^2T/dy^2)
    Discretizada como:
    T_new(i,j) = T_old(i,j) + c * (T_old(i-1,j) + T_old(i+1,j) + T_old(i,j-1) + T_old(i,j+1) - 4*T_old(i,j))
    Onde c = alpha * dt / (dx^2).
    """
    def __init__(self, grid_size, initial_temp, boundary_temp, alpha, dt, dx):
        """
        Inicializa os parâmetros comuns da simulação de difusão de calor.

        Args:
            grid_size (int): Dimensão da grade (e.g., grid_size x grid_size).
                             Deve ser um valor >= 3 para ter células internas.
            initial_temp (float): Temperatura inicial uniforme para a maior parte da grade.
            boundary_temp (float): Temperatura das bordas (condição de contorno de Dirichlet).
            alpha (float): Coeficiente de difusividade térmica (m^2/s).
                           Um valor positivo que determina a rapidez com que o calor se difunde.
            dt (float): Passo de tempo (segundos). Pequenos valores garantem estabilidade.
            dx (float): Espaçamento da grade (metros, assumindo dx=dy para uma grade quadrada).
        
        Raises:
            ValueError: Se os parâmetros de entrada não forem válidos.
            TypeError: Se os tipos dos parâmetros não forem válidos.
        """
        # --- Validação de Parâmetros ---
        if not isinstance(grid_size, int) or grid_size < 3:
            raise ValueError("grid_size deve ser um inteiro maior ou igual a 3 para ter células internas.")
        if not all(isinstance(arg, (int, float)) for arg in [initial_temp, boundary_temp, alpha, dt, dx]):
            raise TypeError("As temperaturas, alpha, dt e dx devem ser números (int ou float).")
        if alpha <= 0 or dt <= 0 or dx <= 0:
            raise ValueError("alpha, dt e dx devem ser valores positivos para uma simulação física válida.")
        
        self.grid_size = grid_size
        self.alpha = alpha
        self.dt = dt
        self.dx = dx
        self.boundary_temp = boundary_temp

        # Fator de atualização para o stencil de 5 pontos.
        # 'c' é um termo constante que otimiza o cálculo em cada célula.
        self.c = alpha * dt / (dx**2)

        # --- Verificação da Condição CFL (Courant-Friedrichs-Lewy) para Estabilidade Numérica ---
        # Para esquemas explícitos 2D da equação do calor, a condição de estabilidade é c <= 0.25.
        # Se esta condição não for atendida, a solução numérica pode divergir e produzir resultados irrealistas.
        if self.c > 0.25:
            print(f"Atenção: A condição CFL ({self.c:.4f} > 0.25) pode levar a instabilidade numérica. "
                  f"Considere diminuir o passo de tempo (dt) ou aumentar o espaçamento da grade (dx).")

        # --- Inicialização das Grades de Temperatura (Double Buffering) ---
        # Usamos duas grades (current_grid e next_grid) para implementar o "double buffering".
        # Isso é essencial para esquemas explícitos, pois a atualização de uma célula
        # deve usar apenas os valores de temperatura do *passo de tempo anterior*.
        # Se atualizássemos a grade in-place, estaríamos usando uma mistura de valores antigos e novos,
        # o que levaria a resultados incorretos e instabilidade.
        # np.float64 é usado para garantir alta precisão nos cálculos de ponto flutuante.
        self.current_grid = np.full((grid_size, grid_size), initial_temp, dtype=np.float64)
        self.next_grid = np.copy(self.current_grid) # next_grid é uma cópia inicial da current_grid

        # Aplica as condições de contorno iniciais a ambas as grades.
        self._apply_boundary_conditions(self.current_grid)
        self._apply_boundary_conditions(self.next_grid)

    def _apply_boundary_conditions(self, grid):
        """
        Aplica as condições de contorno de Dirichlet (temperatura fixa nas bordas).
        Modifica a grade fornecida in-place. As bordas são mantidas a `self.boundary_temp`.
        """
        grid[0, :] = self.boundary_temp  # Borda superior (primeira linha)
        grid[-1, :] = self.boundary_temp # Borda inferior (última linha)
        grid[:, 0] = self.boundary_temp  # Borda esquerda (primeira coluna)
        grid[:, -1] = self.boundary_temp # Borda direita (última coluna)

    def _update_cell(self, r, c, grid_to_read_from):
        """
        Calcula a nova temperatura de uma célula (r, c) usando o stencil de 5 pontos.
        Esta é a discretização central da equação de difusão de calor 2D.
        
        Args:
            r (int): Índice da linha da célula a ser atualizada.
            c (int): Índice da coluna da célula a ser atualizada.
            grid_to_read_from (np.ndarray): A grade de onde ler as temperaturas atuais dos vizinhos.
                                            Para o double buffering, esta é sempre a `self.current_grid`.
                                            
        Returns:
            float: A nova temperatura calculada para a célula (r, c) no próximo passo de tempo.
        """
        # Coleta as temperaturas da célula central e seus quatro vizinhos diretos (Norte, Sul, Leste, Oeste).
        # Estes são os pontos que formam o "stencil" de 5 pontos.
        center = grid_to_read_from[r, c]
        north = grid_to_read_from[r-1, c]
        south = grid_to_read_from[r+1, c]
        east = grid_to_read_from[r, c+1]
        west = grid_to_read_from[r, c-1]

        # Aplica a fórmula do stencil de 5 pontos para a equação do calor 2D explícita.
        # Esta fórmula calcula a mudança de temperatura com base na diferença de temperatura
        # entre a célula central e seus vizinhos.
        return center + self.c * (north + south + east + west - 4 * center)

    def _initialize_simulation_state(self, initial_temp, hotspot_pos, hotspot_temp):
        """
        Reinicializa as grades de temperatura para um novo estado inicial.
        Isso é útil para executar múltiplas simulações (e.g., sequencial e paralela)
        com os mesmos parâmetros iniciais, garantindo que cada run comece do zero.

        Args:
            initial_temp (float): Temperatura inicial uniforme para a grade.
            hotspot_pos (tuple): Posição (linha, coluna) de um ponto quente.
            hotspot_temp (float): Temperatura do ponto quente.
        
        Raises:
            ValueError: Se a posição do hotspot estiver fora das células internas.
        """
        # Preenche ambas as grades com a temperatura inicial uniforme.
        self.current_grid.fill(initial_temp)
        self.next_grid.fill(initial_temp)
        
        # Aplica o ponto quente, se especificado.
        if hotspot_pos:
            # Garante que o hotspot não seja colocado nas bordas, pois as bordas têm temperatura fixa.
            if not (0 < hotspot_pos[0] < self.grid_size - 1 and 0 < hotspot_pos[1] < self.grid_size - 1):
                raise ValueError(f"hotspot_pos {hotspot_pos} deve estar dentro das células internas da grade (não nas bordas).")
            self.current_grid[hotspot_pos] = hotspot_temp
            self.next_grid[hotspot_pos] = hotspot_temp # Mantém o hotspot também na next_grid para consistência
        
        # Aplica as condições de contorno após definir o hotspot.
        self._apply_boundary_conditions(self.current_grid)
        self._apply_boundary_conditions(self.next_grid)
