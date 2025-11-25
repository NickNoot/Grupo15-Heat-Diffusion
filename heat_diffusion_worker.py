
import socket
import numpy as np
import time
import pickle # Importar pickle para tratamento de erros
import struct # Importar struct para tratamento de erros
from shared_utils import send_pickled_data, receive_pickled_data, BaseHeatDiffusion

class HeatDiffusionWorker(BaseHeatDiffusion):
    """
    Um Worker processa uma sub-seção da grade da simulação de difusão de calor.
    Ele recebe os dados do Master (incluindo halos), calcula as novas temperaturas
    para sua região interna e envia os resultados de volta.
    """
    def __init__(self, master_host, master_port):
        # BaseHeatDiffusion será inicializada mais tarde com a configuração do Master.
        # Valores temporários são usados aqui, pois os parâmetros reais vêm do Master.
        super().__init__(grid_size=0, alpha=0, dt=0, dx=0, boundary_temp=0) 
        self.master_host = master_host
        self.master_port = master_port
        self.sock = None # Socket para conexão com o Master

    def _connect_to_master(self):
        """Estabelece a conexão do worker com o Master."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Worker: Tentando conectar a {self.master_host}:{self.master_port}...")
        # Aumentar o timeout de conexão pode ser útil em redes com maior latência.
        self.sock.settimeout(30) # 30 segundos de timeout para conexão
        self.sock.connect((self.master_host, self.master_port))
        self.sock.settimeout(None) # Resetar timeout para operações de leitura/escrita
        print(f"Worker: Conectado ao Master em {self.master_host}:{self.master_port}.")

    def run(self):
        """
        Inicia a operação do worker. Conecta-se ao Master, recebe a configuração inicial,
        e entra em um loop para processar as iterações da simulação.
        """
        self._connect_to_master()

        try:
            # 1. Recebe a configuração inicial do Master.
            # Esta é a primeira mensagem esperada do Master após a conexão.
            initial_config = receive_pickled_data(self.sock)
            if initial_config is None or initial_config.get("type") != "INITIAL_CONFIG":
                raise ValueError("Configuração inicial inválida ou ausente recebida do Master. O worker não pode prosseguir.")
            
            # Atualiza os parâmetros da classe BaseHeatDiffusion com os valores fornecidos pelo Master.
            # Isso garante que o worker use os mesmos parâmetros físicos da simulação.
            self.grid_size = initial_config["grid_size_full"] # A grade completa do Master
            self.alpha = initial_config["alpha"]
            self.dt = initial_config["dt"]
            self.dx = initial_config["dx"]
            self.boundary_temp = initial_config["boundary_temp"]
            # Recalcula 'c' que é o fator de difusão, essencial para a equação da difusão de calor.
            self.c = self.alpha * self.dt / (self.dx**2) 

            print(f"Worker: Configuração recebida. Grade global: {self.grid_size}x{self.grid_size}, dt: {self.dt}, dx: {self.dx}, c: {self.c:.4f}")

            # Loop principal para processar cada iteração da simulação.
            # O worker permanece ativo e processando até que o Master envie uma mensagem de término.
            while True:
                # 2. Recebe os dados da iteração do Master.
                iter_data = receive_pickled_data(self.sock)
                if iter_data is None:
                    # Se receber None, significa que a conexão foi fechada inesperadamente (Master terminou ou falhou).
                    print("Worker: Conexão com Master encerrada inesperadamente. Finalizando.")
                    break
                if iter_data.get("type") == "TERMINATE": # Caso o Master envie uma mensagem explícita de término.
                    print("Worker: Mensagem de término recebida do Master. Finalizando.")
                    break
                if iter_data.get("type") != "ITERATION_UPDATE":
                    raise ValueError(f"Tipo de mensagem inesperado recebido: {iter_data.get('type')}. Esperado 'ITERATION_UPDATE'.")

                # Desempacota os dados recebidos para a iteração atual.
                sub_grid_core = iter_data["sub_grid"] # A parte da grade que este worker é responsável por calcular.
                halo_top = iter_data["halo_top"]       # Linha de dados da grade acima da sub_grid_core.
                halo_bottom = iter_data["halo_bottom"] # Linha de dados da grade abaixo da sub_grid_core.
                hotspot_pos_relative = iter_data["hotspot_pos_relative"] # Posição do hotspot relativa à sub_grid_core.
                hotspot_temp = iter_data["hotspot_temp"]

                # Constrói a grade de trabalho local do worker, incluindo sua sub-grade e os halos.
                # Esta `local_grid` é crucial. Ela permite que o worker aplique o stencil de 5 pontos
                # corretamente, mesmo para as células em suas próprias bordas internas,
                # pois elas terão acesso aos valores dos vizinhos nos halos.
                
                # A dimensão da sub-grade do worker (número de linhas que ele calcula x grid_size).
                num_rows_worker = sub_grid_core.shape[0] 
                
                # `local_grid` é uma grade temporária onde os cálculos serão realizados.
                # Ela tem `num_rows_worker` linhas para o core do worker, mais 2 linhas extras para os halos (top e bottom).
                local_grid = np.zeros((num_rows_worker + 2, self.grid_size), dtype=np.float64)

                # Preenche a `local_grid` com os dados recebidos:
                # A linha 0 da `local_grid` corresponde ao halo superior.
                if halo_top is not None:
                    local_grid[0, :] = halo_top
                else: # Se não há halo superior, significa que esta sub-grade está na borda superior da grade global.
                      # Aplica a condição de contorno global para a linha superior da `local_grid`.
                    local_grid[0, :] = self.boundary_temp

                # As linhas 1 até `num_rows_worker` (inclusive) da `local_grid` correspondem ao core da sub-grade.
                local_grid[1:num_rows_worker + 1, :] = sub_grid_core

                # A última linha (`num_rows_worker + 1`) da `local_grid` corresponde ao halo inferior.
                if halo_bottom is not None:
                    local_grid[num_rows_worker + 1, :] = halo_bottom
                else: # Se não há halo inferior, esta sub-grade está na borda inferior da grade global.
                      # Aplica a condição de contorno global para a linha inferior da `local_grid`.
                    local_grid[num_rows_worker + 1, :] = self.boundary_temp
                
                # `next_local_grid` para double buffering interno do worker.
                # Os resultados da iteração atual serão armazenados aqui.
                # Também tem 2 linhas extras para os halos, mas seus valores de halo
                # não serão atualizados pelo worker, apenas as células internas.
                next_local_grid = np.copy(local_grid)

                # Calcula as novas temperaturas para as células internas da sub-grade do worker.
                # As células a serem atualizadas vão da linha 1 até `num_rows_worker` (inclusive) da `local_grid`.
                # As colunas internas são de 1 a `self.grid_size - 2` (excluindo as bordas laterais que são fixas).
                # Notar que os índices `r_local` e `c_local` são *relativos* à `local_grid`, que inclui os halos.
                for r_local in range(1, num_rows_worker + 1): # Itera sobre as linhas do core do worker
                    for c_local in range(1, self.grid_size - 1): # Itera sobre as colunas internas
                        # Se há um hotspot nesta célula (na sua posição relativa), sua temperatura é fixada.
                        # `r_local - 1` é usado para converter o índice da `local_grid` para o índice relativo dentro da `sub_grid_core`.
                        if hotspot_pos_relative and hotspot_pos_relative[0] == r_local - 1 and hotspot_pos_relative[1] == c_local:
                            next_local_grid[r_local, c_local] = hotspot_temp
                        else:
                            # Chama o método `_update_cell` da classe base para calcular a nova temperatura.
                            # Este método usa os valores dos vizinhos na `local_grid` atual.
                            next_local_grid[r_local, c_local] = self._update_cell(r_local, c_local, local_grid)

                # O worker extrai apenas a sua sub-grade atualizada (sem os halos) para enviar de volta ao Master.
                # O Master é responsável por montar a grade global a partir dessas sub-grades.
                updated_sub_grid_for_master = next_local_grid[1:num_rows_worker + 1, :].copy()

                # 3. Envia a sub-grade atualizada de volta para o Master.
                send_pickled_data(self.sock, {
                    "type": "SUB_GRID_RESULT",
                    "updated_sub_grid": updated_sub_grid_for_master
                })

        except (socket.error, pickle.UnpicklingError, struct.error, ValueError) as e:
            print(f"Worker: Erro durante a execução ou comunicação: {e}")
        finally:
            if self.sock:
                self.sock.close()
                print("Worker: Socket fechado.")

# --- Exemplo de Uso do Worker ---
if __name__ == "__main__":
    # O worker precisa saber onde o Master está escutando para poder se conectar.
    MASTER_HOST = '127.0.0.1' # Deve ser o mesmo IP que o Master está usando.
    MASTER_PORT = 12345       # Deve ser a mesma porta que o Master está usando.

    worker = HeatDiffusionWorker(MASTER_HOST, MASTER_PORT)
    worker.run()