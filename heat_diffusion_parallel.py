
import numpy as np
import threading
import time
import math # Para math.isclose em verificações de resultado, se necessário

from heat_diffusion_base import BaseHeatDiffusion

class ParallelHeatDiffusionSolver(BaseHeatDiffusion):
    """
    Resolve o problema de difusão de calor em paralelo usando threads.
    A sincronização é feita com threading.Barrier para coordenar as iterações
    entre os threads worker e o thread principal.
    """
    def __init__(self, grid_size, initial_temp, boundary_temp, alpha, dt, dx):
        """
        Inicializa o solver paralelo, chamando o construtor da classe base.
        """
        super().__init__(grid_size, initial_temp, boundary_temp, alpha, dt, dx)

    def solve(self, num_iterations, num_threads, hotspot_pos=None, hotspot_temp=None):
        """
        Executa a simulação de difusão de calor em paralelo usando threads.

        Args:
            num_iterations (int): Número total de passos de tempo da simulação.
            num_threads (int): Número de threads a serem utilizados para paralelizar o cálculo.
            hotspot_pos (tuple, optional): Posição (linha, coluna) de um ponto quente fixo.
            hotspot_temp (float, optional): Temperatura do ponto quente.
        
        Returns:
            np.ndarray: A grade final de temperaturas.
        
        Raises:
            ValueError: Se os parâmetros de entrada para threads ou hotspot não forem válidos.
        """
        if not isinstance(num_threads, int) or num_threads <= 0:
            raise ValueError("O número de threads deve ser um inteiro positivo.")
        if (hotspot_pos is not None and hotspot_temp is None) or \
           (hotspot_pos is None and hotspot_temp is not None):
            raise ValueError("hotspot_pos e hotspot_temp devem ser fornecidos juntos ou nenhum deles.")
        
        # --- Otimização do Número de Threads ---
        # O número de linhas internas que precisam ser processadas é (self.grid_size - 2).
        # Não faz sentido ter mais threads do que linhas internas para distribuir,
        # pois threads adicionais ficariam ociosos.
        total_internal_rows = self.grid_size - 2
        if num_threads > total_internal_rows:
            print(f"Atenção: Reduzindo o número de threads de {num_threads} para {total_internal_rows} "
                  f"para evitar threads ociosos, pois há apenas {total_internal_rows} linhas internas para processar.")
            num_threads = max(1, total_internal_rows) # Garante pelo menos 1 thread se grid_size > 2

        # Reinicia o estado da simulação para garantir que cada chamada comece do zero.
        self._initialize_simulation_state(self.current_grid[1,1], hotspot_pos, hotspot_temp)

        # --- Configuração da Barreira de Sincronização ---
        # A barreira (`threading.Barrier`) é um ponto de encontro onde os threads esperam uns pelos outros.
        # Ela garante que todos os threads worker concluam seus cálculos para um passo de tempo
        # antes que qualquer um deles (ou o thread principal) avance para o próximo passo ou troque as grades.
        # O `parties` para a barreira deve incluir todos os threads worker E o thread principal,
        # pois todos precisam sincronizar em cada iteração.
        barrier = threading.Barrier(num_threads + 1) # +1 para o thread principal

        def _worker_iteration_loop(start_r, end_r, barrier_obj):
            """
            Função que cada thread worker executa. Calcula a difusão de calor
            para um conjunto específico de linhas (domínio de trabalho).
            """
            # As grades `self.current_grid` e `self.next_grid` são atributos da instância e,
            # portanto, são compartilhadas entre todos os threads.
            # O Python GIL (Global Interpreter Lock) geralmente limita o paralelismo real
            # para código Python puro. No entanto, operações intensivas em C/Fortran
            # como as de NumPy, liberam o GIL, permitindo que outros threads Python
            # executem código NumPy em paralelo. Isso é crucial para o desempenho aqui.
            for _ in range(num_iterations):
                # Cada worker processa seu intervalo de linhas atribuído.
                for r in range(start_r, end_r):
                    for c in range(1, self.grid_size - 1): # Colunas internas
                        # Se há um hotspot nesta célula, sua temperatura é fixada.
                        # Isso garante que o hotspot seja persistente mesmo se um worker tentar atualizá-lo.
                        if hotspot_pos and hotspot_pos[0] == r and hotspot_pos[1] == c:
                            self.next_grid[r, c] = hotspot_temp
                        else:
                            # Calcula a nova temperatura e armazena na `next_grid`.
                            self.next_grid[r, c] = self._update_cell(r, c, self.current_grid)
                
                # --- Ponto de Sincronização da Barreira ---
                # Cada thread worker espera na barreira. Ele só prosseguirá quando todos os outros
                # workers e o thread principal também tiverem chegado a este ponto.
                # Isso garante que `self.next_grid` esteja completamente preenchida com os novos valores
                # por todos os workers antes que o swap de grades ocorra.
                try:
                    barrier_obj.wait()
                except threading.BrokenBarrierError:
                    # Uma BrokenBarrierError ocorre se um dos threads falhar ou se a barreira for resetada.
                    print(f"Worker thread (linhas {start_r}-{end_r-1}) detectou barreira quebrada. Terminando.")
                    return # Sai do loop de iterações

        threads = []
        # --- Divisão de Domínio (Row-wise Decomposition) ---
        # Divide as linhas internas da grade entre os threads.
        # Cada thread será responsável por um subconjunto contíguo de linhas.
        rows_per_thread = total_internal_rows // num_threads
        remaining_rows = total_internal_rows % num_threads

        current_row_idx = 1 # Começa na primeira linha interna (índice 1, pois a linha 0 é borda)

        # Cria e configura os threads, atribuindo a cada um um intervalo de linhas.
        for i in range(num_threads):
            rows_to_assign = rows_per_thread + (1 if i < remaining_rows else 0) # Distribui as linhas restantes
            end_row_idx = current_row_idx + rows_to_assign
            
            thread = threading.Thread(target=_worker_iteration_loop,
                                      args=(current_row_idx, end_row_idx, barrier))
            threads.append(thread)
            current_row_idx = end_row_idx

        # Inicia todos os threads worker
        for t in threads:
            t.start()

        # --- Loop Principal do Thread Master (Coordenador) ---
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            # O thread principal também espera na barreira. Ele aguarda que todos os workers
            # tenham terminado de calcular suas partes da `next_grid` para o passo de tempo atual.
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                print("Main thread detectou barreira quebrada. Terminando simulação.")
                break # Sai do loop de iterações

            # Após a sincronização, o thread principal realiza a troca de grades (double buffering).
            # Isso é feito pelo thread principal para evitar condições de corrida entre os workers
            # ao manipular as referências das grades.
            self.current_grid, self.next_grid = self.next_grid, self.current_grid

            # Reaplica as condições de contorno na nova `current_grid` (que era a `next_grid` anterior).
            # Isso garante que as bordas permaneçam fixas.
            self._apply_boundary_conditions(self.current_grid)

            # Mantém a temperatura do hotspot constante.
            if hotspot_pos:
                self.current

            # Mantém a temperatura do hotspot constante.
            # Esta operação é segura para ser feita pelo thread principal, pois não interfere
            # com as regiões da grade que os worker threads estão calculando.
            if hotspot_pos:
                self.current_grid[hotspot_pos] = hotspot_temp
        
        # O thread principal espera que todos os worker threads terminem suas execuções.
        # Eles devem terminar naturalmente após a última chamada de barrier.wait() e a conclusão do loop principal.
        # Isso garante que todos os recursos dos threads sejam liberados corretamente.
        for t in threads:
            t.join()

        end_time = time.perf_counter()
        print(f"Tempo de execução paralelo ({num_threads} threads): {end_time - start_time:.4f} segundos")
        return self.current_grid

# --- Exemplo de Uso do Solver Paralelo ---
if __name__ == "__main__":
    # --- Parâmetros da Simulação ---
    GRID_SIZE = 200        # Dimensão da grade (e.g., 200x200 células). Uma grade maior realça os benefícios do paralelismo.
    INITIAL_TEMP = 20.0    # Temperatura inicial uniforme de toda a grade.
    BOUNDARY_TEMP = 0.0    # Temperatura fixa nas bordas (condição de contorno de Dirichlet).
    HOTSPOT_TEMP = 100.0   # Temperatura do ponto quente, mantida constante.
    HOTSPOT_POS = (GRID_SIZE // 2, GRID_SIZE // 2) # Posição central do ponto quente.
    ALPHA = 0.1            # Coeficiente de difusividade térmica do material.
                           # Representa quão rapidamente o calor se espalha.
    DX = 1.0               # Espaçamento da grade (distância entre os centros das células).
                           # Assume-se DX = DY para simplificação.
    DT = 0.1               # Passo de tempo para cada iteração da simulação.
                           # A escolha de DT é crítica para a estabilidade numérica do método explícito.
                           # Para estabilidade, a condição de Courant-Friedrichs-Lewy (CFL) deve ser satisfeita:
                           # DT <= DX**2 / (4 * ALPHA) para 2D.
                           # Neste exemplo, 0.1 <= 1**2 / (4 * 0.1) => 0.1 <= 1 / 0.4 => 0.1 <= 2.5, o que é estável.
    NUM_ITERATIONS = 500   # Número total de passos de tempo para a simulação.
    
    # --- Configuração do Paralelismo ---
    # Experimente com diferentes números de threads (e.g., 2, 4, 8) para observar o speedup.
    # O número ideal de threads é geralmente próximo ao número de núcleos lógicos da CPU para cargas de trabalho intensivas em CPU.
    # Cuidado para não definir um NUM_THREADS muito alto para grades pequenas,
    # pois pode haver mais overhead de gerenciamento de threads (criação, sincronização, troca de contexto)
    # do que ganho computacional, resultando em desempenho pior que o sequencial.
    NUM_THREADS = 4        

    print(f"\n--- Simulação Paralela de Difusão de Calor ---")
    print(f"Grade: {GRID_SIZE}x{GRID_SIZE}, Iterações: {NUM_ITERATIONS}, Threads: {NUM_THREADS}")
    print(f"Parâmetros: Alpha={ALPHA}, DT={DT}, DX={DX}")

    # Instancia o solver paralelo
    solver_parallel = ParallelHeatDiffusionSolver(GRID_SIZE, INITIAL_TEMP, BOUNDARY_TEMP, ALPHA, DT, DX)
    # Executa a simulação
    final_grid_parallel = solver_parallel.solve(NUM_ITERATIONS, NUM_THREADS, HOTSPOT_POS, HOTSPOT_TEMP)

    # --- Visualização dos Resultados (Opcional) ---
    try:
        import matplotlib.pyplot as plt
        # Cria uma figura para o plot
        plt.figure(figsize=(9, 7))
        # Exibe a grade de temperatura como uma imagem.
        # 'cmap='hot'' usa um mapa de cores que vai do preto (frio) ao amarelo/branco (quente).
        # 'origin='lower'' garante que o índice [0,0] da matriz corresponda ao canto inferior esquerdo do gráfico,
        # o que é comum em gráficos cartesianos e evita a inversão vertical da imagem.
        # 'vmin' e 'vmax' fixam a escala de cores, tornando-a consistente e interpretável.
        plt.imshow(final_grid_parallel, cmap='hot', origin='lower', vmin=BOUNDARY_TEMP, vmax=HOTSPOT_TEMP)
        plt.colorbar(label='Temperatura (°C)') # Adiciona uma barra de cores para indicar a escala de temperatura.
        plt.title(f'Difusão de Calor Paralela ({NUM_THREADS} Threads) após {NUM_ITERATIONS} Iterações')
        plt.xlabel('Posição X (células)')
        plt.ylabel('Posição Y (células)')
        plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos.
        plt.show()
    except ImportError:
        print("\nMatplotlib não está instalado. Para visualização, instale com 'pip install matplotlib'.")
    except Exception as e:
        print(f"\nErro inesperado ao tentar plotar os resultados: {e}")
