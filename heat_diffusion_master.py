
import socket
import threading
import numpy as np
import time
import math
from shared_utils import send_pickled_data, receive_pickled_data, BaseHeatDiffusion

class HeatDiffusionMaster(BaseHeatDiffusion):
    """
    O Master coordena a simulação de difusão de calor distribuída.
    Ele gerencia a grade global, particiona o trabalho entre os workers,
    distribui as sub-grades e halos, coleta os resultados e sincroniza as iterações.
    Utiliza double buffering para atualizar a grade global de forma segura e eficiente.
    """
    def __init__(self, host, port, grid_size, initial_temp, boundary_temp,
                 alpha, dt, dx, num_workers, hotspot_pos, hotspot_temp):
        # Inicializa a classe base com os parâmetros da simulação
        super().__init__(grid_size, alpha, dt, dx, boundary_temp)
        
        self.host = host
        self.port = port
        self.initial_temp = initial_temp
        self.hotspot_pos = hotspot_pos
        self.hotspot_temp = hotspot_temp
        self.num_workers = num_workers

        # Armazena as conexões dos workers (socket, endereço)
        self.connections = [] 
        # Armazena dados específicos de cada worker, como faixas de linhas
        self.worker_info = {} 

        # Grade global atual e próxima grade global para double buffering.
        # Double buffering é essencial para garantir que todos os cálculos da iteração atual
        # usem os valores da grade do tempo 't' antes que qualquer parte dela seja atualizada
        # para o tempo 't+1'.
        self.current_global_grid = np.full((grid_size, grid_size), initial_temp, dtype=np.float64)
        self.next_global_grid = np.copy(self.current_global_grid)

        # Aplica as condições iniciais (hotspot e bordas) à grade global
        if self.hotspot_pos:
            # Assegura que o hotspot é fixo na grade global
            self.current_global_grid[self.hotspot_pos] = self.hotspot_temp
            self.next_global_grid[self.hotspot_pos] = self.hotspot_temp
        self._apply_boundary_conditions(self.current_global_grid, self.boundary_temp)
        self._apply_boundary_conditions(self.next_global_grid, self.boundary_temp)

        # Barreira de sincronização para todos os worker threads e o thread principal do Master.
        # Garante que todos completem uma iteração antes de prosseguir para a próxima.
        # O +1 é para incluir o thread principal do Master na sincronização.
        self.iteration_barrier = threading.Barrier(num_workers + 1) 

    def _partition_grid(self):
        """
        Calcula as faixas de linhas da grade global que cada worker será responsável.
        A partição é feita apenas nas linhas *internas* da grade global,
        excluindo a primeira e a última linha que são bordas fixas globais.
        A distribuição tenta ser o mais uniforme possível.
        """
        # Número total de linhas internas (exclui a primeira e a última linha que são bordas fixas)
        total_internal_rows = self.grid_size - 2 
        
        if total_internal_rows < self.num_workers:
            raise ValueError(f"Número de workers ({self.num_workers}) excede o número de linhas internas ({total_internal_rows}). "
                             "Ajuste GRID_SIZE ou NUM_WORKERS.")

        rows_per_worker = total_internal_rows // self.num_workers
        remainder = total_internal_rows % self.num_workers
        
        current_global_row = 1 # Começa após a borda superior (linha global de índice 0)
        
        for i in range(self.num_workers):
            start_row = current_global_row
            # Adiciona 1 linha extra para os primeiros 'remainder' workers para distribuir as linhas restantes
            end_row = start_row + rows_per_worker + (1 if i < remainder else 0)
            
            # Armazena a faixa de linhas globais (exclusivo para end_row)
            self.worker_info[i] = {
                "id": i,
                "start_global_row": start_row, # Primeira linha global (inclusive)
                "end_global_row": end_row,     # Última linha global (exclusive)
                "socket": None, # Será preenchido após a conexão
                "addr": None
            }
            current_global_row = end_row
        
    def _worker_handler_thread(self, conn, addr, worker_id):
        """
        Thread dedicada a lidar com a comunicação com um worker específico.
        Gerencia o envio de dados de iteração e o recebimento de resultados.
        Cada worker tem seu próprio thread de handler no Master para comunicação paralela.
        """
        self.worker_info[worker_id]["socket"] = conn
        self.worker_info[worker_id]["addr"] = addr

        print(f"Master: Worker {worker_id} conectado de {addr}. Designado às linhas globais "
              f"[{self.worker_info[worker_id]['start_global_row']}:{self.worker_info[worker_id]['end_global_row']-1}]")

        try:
            # 1. Envia a configuração inicial ao worker
            config = {
                "type": "INITIAL_CONFIG",
                "grid_size_full": self.grid_size,
                "alpha": self.alpha,
                "dt": self.dt,
                "dx": self.dx,
                "boundary_temp": self.boundary_temp,
            }
            send_pickled_data(conn, config)

            # 2. Loop de iterações para este worker
            # `self.num_iterations_total` é definido no método `run()`
            for _ in range(self.num_iterations_total): 
                # Prepara os dados para o worker: sua sub-grade e as regiões de halo
                start_r = self.worker_info[worker_id]['start_global_row']
                end_r = self.worker_info[worker_id]['end_global_row']

                # Extrai a sub-grade do worker da grade global atual.
                # O `.copy()` é importante para evitar que o worker modifique diretamente
                # a grade global do master antes da etapa de double buffering.
                sub_grid_core = self.current_global_grid[start_r:end_r, :].copy()
                
                # Extrai a linha de halo superior.
                # Esta é a linha imediatamente acima da sub-grade do worker na grade global.
                # É necessária para o cálculo da primeira linha da sub-grade do worker.
                halo_top = None
                if start_r > 0: # Se não for a primeira linha interna global (que tem a borda 0 como halo)
                    halo_top = self.current_global_grid[start_r - 1, :].copy()
                
                # Extrai a linha de halo inferior.
                # Esta é a linha imediatamente abaixo da sub-grade do worker na grade global.
                # É necessária para o cálculo da última linha da sub-grade do worker.
                halo_bottom = None
                if end_r < self.grid_size: # Se não for a última linha interna global (que tem a borda N-1 como halo)
                    halo_bottom = self.current_global_grid[end_r, :].copy()
                
                # Verifica se o hotspot está na área deste worker e calcula sua posição relativa.
                # O hotspot é uma condição de contorno interna fixa.
                hotspot_pos_relative = None
                if self.hotspot_pos and start_r <= self.hotspot_pos[0] < end_r:
                    # A posição relativa é a linha do hotspot dentro da sub_grid_core do worker.
                    hotspot_pos_relative = (self.hotspot_pos[0] - start_r, self.hotspot_pos[1])

                # Envia os dados da iteração para o worker
                iter_data = {
                    "type": "ITERATION_UPDATE",
                    "sub_grid": sub_grid_core,
                    "halo_top": halo_top,
                    "halo_bottom": halo_bottom,
                    "hotspot_pos_relative": hotspot_pos_relative,
                    "hotspot_temp": self.hotspot_temp
                }
                send_pickled_data(conn, iter_data)

                # Recebe a sub-grade atualizada do worker
                response = receive_pickled_data(conn)
                if response is None or response["type"] != "SUB_GRID_RESULT":
                    print(f"Master: Worker {worker_id} desconectado ou enviou resposta inválida. Encerrando thread.")
                    # Se um worker falhar, a barreira será quebrada e a simulação principal encerrada.
                    self.iteration_barrier.abort() 
                    break 
                
                updated_sub_grid = response["updated_sub_grid"]
                # Atualiza a parte correspondente na próxima grade global do Master.
                # Esta é a parte do double buffering.
                self.next_global_grid[start_r:end_r, :] = updated_sub_grid
                
                # Sincroniza com o thread principal do Master e outros workers via barreira.
                # Todos os worker_handler_threads e o thread principal devem chegar aqui
                # antes que a próxima iteração possa começar.
                self.iteration_barrier.wait() 

        except (socket.error, pickle.UnpicklingError, struct.error, threading.BrokenBarrierError) as e:
            print(f"Master: Erro na comunicação com Worker {worker_id} em {addr}: {e}")
            # Em caso de erro, aborta a barreira para evitar que outros threads fiquem bloqueados.
            self.iteration_barrier.abort()
        finally:
            # Garante que, se o worker handler falhar, ele sinalize a barreira como quebrada
            # para que outros threads não fiquem bloqueados indefinidamente.
            # A verificação `n_waiting > 0` é uma otimização, mas `abort()` é seguro mesmo se ninguém estiver esperando.
            if self.iteration_barrier.n_waiting > 0: 
                self.iteration_barrier.abort()
            conn.close()
            print(f"Master: Conexão com Worker {worker_id} ({addr}) encerrada.")

    def run(self, num_iterations):
        """
        Executa a simulação distribuída.
        Configura o servidor, aceita conexões de workers e gerencia o loop principal de iterações.
        """
        self.num_iterations_total = num_iterations # Armazena o número total de iterações
        try:
            self._partition_grid() # Calcula as partições da grade
        except ValueError as e:
            print(f"Master: Erro de configuração: {e}")
            return None

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR permite que o socket seja reutilizado imediatamente após o fechamento,
        # evitando o erro "Address already in use" em reinícios rápidos.
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        server_socket.bind((self.host, self.port))
        server_socket.listen(self.num_workers) # Limita o número de conexões pendentes
        print(f"Master: Escutando em {self.host}:{self.port} por {self.num_workers} workers...")

        # Aceita conexões de workers e inicia um thread para cada um
        worker_threads = []
        for i in range(self.num_workers):
            try:
                conn, addr = server_socket.accept()
                # Define o thread como daemon. Isso significa que o programa principal
                # pode sair mesmo que esses threads ainda estejam em execução,
                # o que é útil para garantir o encerramento limpo.
                worker_thread = threading.Thread(target=self._worker_handler_thread, args=(conn, addr, i))
                worker_thread.daemon = True 
                worker_thread.start()
                worker_threads.append(worker_thread)
                self.connections.append((conn, addr))
            except socket.error as e:
                print(f"Master: Erro ao aceitar conexão de worker: {e}. Encerrando.")
                server_socket.close()
                return None
        
        print("Master: Todos os workers conectados. Iniciando simulação...")
        start_time = time.perf_counter()

        # Loop principal do Master para gerenciar as iterações
        for iteration in range(num_iterations):
            try:
                # O thread principal do Master espera na barreira até que todos os
                # worker_handler_threads tenham concluído sua parte desta iteração
                self.iteration_barrier.wait() 
            except threading.BrokenBarrierError:
                print("Master: Barreira quebrada. Provável desconexão de worker ou erro. Encerrando simulação.")
                break # Sai do loop principal de iterações

            # Após a sincronização, Master realiza a troca de grades (double buffering).
            # A grade 'next' se torna a 'current' para a próxima iteração.
            self.current_global_grid, self.next_global_grid = self.next_global_grid, self.current_global_grid

            # Reaplica as condições de contorno globais (as bordas são fixas)
            self._apply_boundary_conditions(self.current_global_grid, self.boundary_temp)

            # Reaplica a temperatura do hotspot global (se houver)
            # O Master é o único responsável por manter a consistência do hotspot na grade global.
            if self.hotspot_pos:
                self.current_global_grid[self.hotspot_pos] = self.hotspot_temp
            
            # Não precisamos resetar self.next_global_grid aqui, pois os workers o preencherão na próxima iteração
            # com base na nova self.current_global_grid. A troca de buffers no início da próxima iteração
            # fará com que self.next_global_grid se torne a base para os novos cálculos.

            # Opcional: imprimir progresso para feedback ao usuário
            if (iteration + 1) % (num_iterations // 10 if num_iterations >= 10 else 1) == 0 or iteration == num_iterations -1:
                 print(f"Master: Iteração {iteration + 1}/{num_iterations} concluída.")

        end_time = time.perf_counter()
        print(f"Master: Simulação distribuída concluída em {end_time - start_time:.4f} segundos.")
        
        # Garante que todos os worker threads terminem (caso não sejam daemons ou para esperar por logs finais)
        # É uma boa prática aguardar a conclusão de todos os threads para evitar vazamentos de recursos.
        for t in worker_threads:
            t.join()

        # Envia uma mensagem explícita de término para cada worker para que eles possam encerrar graciosamente.
        # Isso é importante para evitar que os workers fiquem bloqueados esperando por dados que nunca virão.
        for conn, _ in self.connections:
            try:
                send_pickled_data(conn, {"type": "TERMINATE"})
            except socket.error as e:
                print(f"Master: Erro ao enviar mensagem de término para worker: {e}")

        server_socket.close()
        print("Master: Servidor encerrado.")
        return self.current_global_grid

# --- Exemplo de Uso do Master ---
if __name__ == "__main__":
    # Parâmetros da simulação (ajuste conforme necessário para testar escalabilidade e desempenho)
    GRID_SIZE = 200        # Dimensão da grade (e.g., 200x200 células). Grades maiores exigem mais cálculo e comunicação.
    INITIAL_TEMP = 20.0    # Temperatura inicial uniforme de toda a grade.
    BOUNDARY_TEMP = 0.0    # Temperatura fixa nas bordas (condição de Dirichlet). Representa um dissipador de calor.
    HOTSPOT_TEMP = 100.0   # Temperatura de um ponto quente central, mantida constante.
    HOTSPOT_POS = (GRID_SIZE // 2, GRID_SIZE // 2) # Posição do ponto quente (linha, coluna).
    ALPHA = 0.1            # Coeficiente de difusividade térmica (m^2/s). Afeta a velocidade da propagação do calor.
    DX = 1.0               # Espaçamento da grade (m). Tamanho de cada célula.
    DT = 0.1               # Passo de tempo (s). Tamanho do intervalo de tempo para cada iteração.
    NUM_ITERATIONS = 500   # Número de iterações de tempo. Determina a duração da simulação.
    NUM_WORKERS = 4        # Número de processos worker que o Master espera conectar. Essencial para o paralelismo.

    MASTER_HOST = '127.0.0.1' # IP do Master (usar localhost para testes locais). Para redes, use o IP da máquina do Master.
    MASTER_PORT = 12345       # Porta para comunicação com workers. Deve ser uma porta livre.

    master = HeatDiffusionMaster(
        MASTER_HOST, MASTER_PORT, GRID_SIZE, INITIAL_TEMP, BOUNDARY_TEMP,
        ALPHA, DT, DX, NUM_WORKERS, HOTSPOT_POS, HOTSPOT_TEMP
    )

    print(f"--- Simulação Distribuída {GRID_SIZE}x{GRID_SIZE}, {NUM_ITERATIONS} iterações, {NUM_WORKERS} workers ---")
    final_grid_distributed = master.run(NUM_ITERATIONS)

    if final_grid_distributed is not None:
        print("\nSimulação distribuída concluída com sucesso. Resultado final disponível.")
        # Opcional: Salvar a grade final ou visualizar para análise posterior.
        # np.save('final_grid_distributed.npy', final_grid_distributed)

        # Para comparação com a versão sequencial (se desejar executar aqui também)
        # É altamente recomendado para validar a correção do algoritmo distribuído e medir o speedup.
        # from heat_diffusion_parallel import HeatDiffusionSolver # Assumindo que a classe Sequential está lá
        # seq_solver = HeatDiffusionSolver(GRID_SIZE, INITIAL_TEMP, BOUNDARY_TEMP, ALPHA, DT, DX)
        # sequential_result_grid = seq_solver.sequential_solve(NUM_ITERATIONS, HOTSPOT_POS, HOTSPOT_TEMP)

        # if np.allclose(sequential_result_grid, final_grid_distributed, atol=1e-8):
        #     print("Resultados sequencial e distribuído são consistentes! ✅")
        # else:
        #     print("AVISO: Resultados sequencial e distribuído NÃO são idênticos! ❌")
        #     # Pode ser útil imprimir a diferença máxima: np.max(np.abs(sequential_result_grid - final_grid_distributed))

        # Visualização (requer matplotlib para exibir o mapa de calor)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            # `cmap='hot'` para um mapa de calor, `origin='lower'` para que o (0,0) fique no canto inferior esquerdo.
            # `vmin` e `vmax` ajudam a normalizar a escala de cores para melhor visualização da distribuição de temperatura.
            plt.imshow(final_grid_distributed, cmap='hot', origin='lower', vmin=BOUNDARY_TEMP, vmax=HOTSPOT_TEMP)
            plt.colorbar(label='Temperatura (°C)')
            plt.title(f'Difusão de Calor Distribuída ({NUM_WORKERS} Workers)')
            plt.xlabel('Posição X')
            plt.ylabel('Posição Y')
            plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos.
            plt.show()
        except ImportError:
            print("\nMatplotlib não está instalado. Para visualização, instale com 'pip install matplotlib'.")
        except Exception as e:
            print(f"\nErro ao tentar plotar os resultados: {e}")