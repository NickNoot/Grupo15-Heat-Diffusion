
import socket
import pickle
import struct # Para empacotar/desempacotar o comprimento da mensagem
import numpy as np

# --- Utilidades de Comunicação ---
def send_pickled_data(sock, data):
    """
    Serializa dados Python usando pickle e os envia através de um socket TCP.
    A mensagem é precedida por um prefixo de 4 bytes que indica o seu comprimento total.
    Isso garante que o receptor saiba exatamente quantos bytes esperar para a mensagem completa.
    
    Args:
        sock (socket.socket): O objeto socket conectado.
        data (any): Os dados Python a serem enviados.
    
    Raises:
        socket.error: Se ocorrer um erro durante a operação de envio.
    """
    serialized_data = pickle.dumps(data)
    # '!I' significa: network byte order (big-endian), unsigned int (4 bytes)
    length_prefix = struct.pack("!I", len(serialized_data)) 
    try:
        # sendall garante que todos os bytes são enviados ou levanta uma exceção
        sock.sendall(length_prefix + serialized_data)
    except socket.error as e:
        print(f"Erro ao enviar dados: {e}")
        raise # Propaga o erro para ser tratado no Master/Worker

def receive_pickled_data(sock):
    """
    Recebe dados serializados de um socket TCP, lendo primeiro o prefixo de comprimento.
    Isso permite a reconstrução correta da mensagem completa, mesmo que ela chegue em pacotes fragmentados.
    
    Args:
        sock (socket.socket): O objeto socket conectado.
        
    Returns:
        any: Os dados Python desserializados. Retorna None se a conexão for fechada
             ou se ocorrer um erro na leitura do prefixo.
             
    Raises:
        socket.error: Se ocorrer um erro durante a operação de recebimento.
        EOFError: Se a conexão for fechada inesperadamente antes de receber todos os dados.
        pickle.UnpicklingError: Se os dados recebidos não puderem ser desserializados.
        struct.error: Se o prefixo de comprimento for inválido.
    """
    try:
        # Recebe o prefixo de comprimento (4 bytes)
        length_prefix = sock.recv(4, socket.MSG_WAITALL) # MSG_WAITALL garante que todos os 4 bytes sejam recebidos
        if not length_prefix:
            return None # Conexão fechada ou erro antes de receber o prefixo
        if len(length_prefix) < 4: # Pode acontecer se a conexão fechar no meio
            raise EOFError("Conexão fechada inesperadamente ao receber prefixo de comprimento.")
        
        length = struct.unpack("!I", length_prefix)[0] # Desempacota o comprimento
        
        # Recebe os dados reais em blocos até que todos os 'length' bytes sejam recebidos
        # Usar MSG_WAITALL é mais simples e robusto para este caso do que um loop manual.
        data_buffer = sock.recv(length, socket.MSG_WAITALL)
        if not data_buffer:
            raise EOFError("Conexão fechada inesperadamente ao receber dados.")
        if len(data_buffer) < length:
            raise EOFError("Dados incompletos recebidos.")
            
        return pickle.loads(data_buffer) # Desserializa os dados
    except (socket.error, EOFError, pickle.UnpicklingError, struct.error) as e:
        print(f"Erro ao receber ou desserializar dados: {e}")
        raise # Propaga o erro

# --- Lógica Central da Simulação de Difusão de Calor (reutilizável) ---
class BaseHeatDiffusion:
    """
    Classe base contendo a lógica comum para a simulação de difusão de calor,
    reutilizada pelo Master e pelos Workers. Implementa o método de diferenças finitas
    para a equação do calor 2D.
    """
    def __init__(self, grid_size, alpha, dt, dx, boundary_temp):
        self.grid_size = grid_size
        self.alpha = alpha
        self.dt = dt
        self.dx = dx
        self.boundary_temp = boundary_temp
        # Termo constante para a equação de diferenças finitas discretizada.
        # Pré-calculado para eficiência, pois é usado em cada célula a cada iteração.
        self.c = alpha * dt / (dx**2)

        # Aviso sobre a condição CFL (Courant-Friedrichs-Lewy) para estabilidade numérica.
        # Para a equação do calor 2D com o método explícito, c <= 0.25 é necessário para estabilidade.
        if self.c > 0.25:
            print(f"Atenção: A condição CFL (c = {self.c:.4f} > 0.25) pode levar a instabilidade numérica. "
                  f"Considere diminuir o passo de tempo (dt) ou aumentar o espaçamento da grade (dx).")

    def _update_cell(self, r_local, c_local, grid_with_halo):
        """
        Calcula a nova temperatura de uma célula (r_local, c_local) usando o stencil de 5 pontos.
        Este método implementa a discretização da equação do calor 2D usando diferenças finitas explícitas.
        
        Args:
            r_local (int): Índice da linha da célula dentro de `grid_with_halo`.
            c_local (int): Índice da coluna da célula dentro de `grid_with_halo`.
            grid_with_halo (np.ndarray): A grade local que inclui a sub-grade do worker
                                         e, opcionalmente, as regiões de halo.
                                         
        Returns:
            float: A nova temperatura calculada para a célula.
        """
        center = grid_with_halo[r_local, c_local]
        north = grid_with_halo[r_local - 1, c_local]
        south = grid_with_halo[r_local + 1, c_local]
        east = grid_with_halo[r_local, c_local + 1]
        west = grid_with_halo[r_local, c_local - 1]

        # Fórmula do stencil de 5 pontos para a atualização da temperatura:
        # T_new(i,j) = T_old(i,j) + c * (T_old(i-1,j) + T_old(i+1,j) + T_old(i,j-1) + T_old(i,j+1) - 4 * T_old(i,j))
        return center + self.c * (north + south + east + west - 4 * center)

    def _apply_boundary_conditions(self, grid_to_modify, boundary_val):
        """
        Aplica as condições de contorno de Dirichlet (temperatura fixa) a uma grade.
        As células nas bordas da grade são definidas para um valor constante.
        Modifica a grade in-place.
        
        Args:
            grid_to_modify (np.ndarray): A grade NumPy à qual as condições de contorno serão aplicadas.
            boundary_val (float): O valor da temperatura a ser aplicado nas bordas.
        """
        grid_to_modify[0, :] = boundary_val  # Borda superior (linha 0)
        grid_to_modify[-1, :] = boundary_val # Borda inferior (última linha)
        grid_to_modify[:, 0] = boundary_val  # Borda esquerda (coluna 0)
        grid_to_modify[:, -1] = boundary_val # Borda direita (última coluna)
