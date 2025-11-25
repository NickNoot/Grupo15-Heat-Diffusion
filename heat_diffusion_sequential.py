
import numpy as np
import time
import math # Para math.isclose em verificações de resultado, se necessário

from heat_diffusion_base import BaseHeatDiffusion

class SequentialHeatDiffusionSolver(BaseHeatDiffusion):
    """
    Resolve o problema de difusão de calor sequencialmente.
    Esta implementação serve como linha de base para comparação de desempenho
    e validação da correção das implementações paralelas.
    """
    def __init__(self, grid_size, initial_temp, boundary_temp, alpha, dt, dx):
        """
        Inicializa o solver sequencial, chamando o construtor da classe base.
        """
        super().__init__(grid_size, initial_temp, boundary_temp, alpha, dt, dx)

    def solve(self, num_iterations, hotspot_pos=None, hotspot_temp=None):
        """
        Executa a simulação de difusão de calor sequencialmente por um dado número de iterações.

        Args:
            num_iterations (int): Número total de passos de tempo da simulação.
            hotspot_pos (tuple, optional): Posição (linha, coluna) de um ponto quente fixo.
                                           Se fornecido, a temperatura nesta célula será mantida constante.
            hotspot_temp (float, optional): Temperatura do ponto quente. Necessário se hotspot_pos for fornecido.
        
        Returns:
            np.ndarray: A grade final de temperaturas após todas as iterações.
        
        Raises:
            ValueError: Se hotspot_pos for fornecido sem hotspot_temp, ou vice-versa.
        """
        if (hotspot_pos is not None and hotspot_temp is None) or \
           (hotspot_pos is None and hotspot_temp is not None):
            raise ValueError("hotspot_pos e hotspot_temp devem ser fornecidos juntos ou nenhum deles.")

        # Reinicia o estado da simulação para garantir que cada chamada a `solve` comece do zero.
        # Isso é crucial para comparações justas de desempenho ou para executar múltiplas simulações.
        self._initialize_simulation_state(self.current_grid[1,1], hotspot_pos, hotspot_temp)
        # Nota: current_grid[1,1] é usado como uma célula interna representativa para initial_temp.

        start_time = time.perf_counter() # Usa time.perf_counter para medições de tempo de alta precisão.

        for _ in range(num_iterations):
            # Percorre apenas as células internas da grade (exclui as bordas).
            # As bordas são fixas pelas condições de contorno e não são atualizadas pelo stencil.
            # O loop vai de 1 a grid_size-2 (inclusive) para linhas e colunas.
            for r in range(1, self.grid_size - 1):
                for c in range(1, self.grid_size - 1):
                    # Se houver um hotspot nesta célula, sua temperatura é mantida constante e não é atualizada.
                    if hotspot_pos and hotspot_pos[0] == r and hotspot_pos[1] == c:
                        self.next_grid[r, c] = hotspot_temp
                    else:
                        # Calcula a nova temperatura da célula (r, c) usando a grade atual (current_grid)
                        # e armazena o resultado na grade futura (next_grid).
                        self.next_grid[r, c] = self._update_cell(r, c, self.current_grid)

            # --- Double Buffering Swap ---
            # Após todas as células internas terem sido calculadas para o próximo passo de tempo
            # e armazenadas em `self.next_grid`, as grades são trocadas.
            # `self.next_grid` (que agora contém os novos valores) torna-se `self.current_grid`
            # para a próxima iteração, e `self.current_grid` (os valores antigos) torna-se `self.next_grid`
            # para ser preenchida na próxima iteração.
            self.current_grid, self.next_grid = self.next_grid, self.current_grid

            # Garante que as bordas da nova `current_grid` (que era a `next_grid` anterior)
            # permaneçam com as condições de contorno. Embora o `_update_cell` não as toque,
            # esta chamada reforça a imutabilidade das bordas.
            self._apply_boundary_conditions(self.current_grid)

            # Se houver um hotspot, garante que sua temperatura seja restaurada, caso tenha sido
            # acidentalmente sobrescrita ou para reforçar sua constância.
            if hotspot_pos:
                self.current_grid[hotspot_pos] = hotspot_temp

        end_time = time.perf_counter()
        print(f"Tempo de execução sequencial: {end_time - start_time:.4f} segundos")
        return self.current_grid

# --- Exemplo de Uso do Solver Sequencial ---
if __name__ == "__main__":
    # --- Parâmetros da Simulação ---
    GRID_SIZE = 200        # Dimensão da grade (e.g., 200x200 células). Um tamanho maior aumenta o custo computacional.
    INITIAL_TEMP = 20.0    # Temperatura inicial uniforme em toda a grade.
    BOUNDARY_TEMP = 0.0    # Temperatura fixa nas 4 bordas da grade (condição de Dirichlet).
    HOTSPOT_TEMP = 100.0   # Temperatura do ponto quente central.
    HOTSPOT_POS = (GRID_SIZE // 2, GRID_SIZE // 2) # Posição exata do ponto quente.
    ALPHA = 0.1            # Coeficiente de difusividade térmica do material.
    DX = 1.0               # Espaçamento da grade em metros.
    DT = 0.1               # Passo de tempo em segundos. Cuidado com a condição CFL!
    NUM_ITERATIONS = 500   # Número de passos de tempo para simular a difusão.

    print(f"\n--- Simulação Sequencial de Difusão de Calor ---")
    print(f"Grade: {GRID_SIZE}x{GRID_SIZE}, Iterações: {NUM_ITERATIONS}")

    # Instancia o solver sequencial
    solver_seq = SequentialHeatDiffusionSolver(GRID_SIZE, INITIAL_TEMP, BOUNDARY_TEMP, ALPHA, DT, DX)
    # Executa a simulação
    final_grid_seq = solver_seq.solve(NUM_ITERATIONS, HOTSPOT_POS, HOTSPOT_TEMP)

    # --- Visualização dos Resultados (Opcional) ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 7)) # Aumenta o tamanho da figura para melhor visualização
        # imshow plota a grade como uma imagem. 'hot' é um mapa de cores comum para temperatura.
        # 'origin='lower'' faz com que o (0,0) seja o canto inferior esquerdo, como em gráficos cartesianos.
        plt.imshow(final_grid_seq, cmap='hot', origin='lower', vmin=BOUNDARY_TEMP, vmax=HOTSPOT_TEMP)
        plt.colorbar(label='Temperatura (°C)') # Adiciona uma barra de cores para indicar os valores.
        plt.title('Difusão de Calor Sequencial')
        plt.xlabel('Posição X (células)')
        plt.ylabel('Posição Y (células)')
        plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos.
        plt.show()
    except ImportError:
        print("\nMatplotlib não está instalado. Para visualização, instale com 'pip install matplotlib'.")
    except Exception as e:
        print(f"\nErro ao tentar plotar os resultados: {e}")
