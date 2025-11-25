# Simulação de Difusão de Calor: Sequencial, Paralela e Distribuída em Python

Este repositório contém implementações de um modelo clássico de simulação, a Difusão de Calor (Heat Diffusion), explorando diferentes paradigmas de computação: sequencial, paralela (usando threads) e distribuída (usando sockets). O objetivo é comparar os tempos de execução, discutir a escalabilidade e eficiência de cada abordagem, e identificar potenciais melhorias.

## Estrutura de Arquivos

Aqui está uma breve descrição dos arquivos incluídos neste repositório:

-   `heat_diffusion_base.py`:
    -   Contém a lógica fundamental e comum para a simulação de difusão de calor. Define a estrutura básica da grade, a aplicação de condições de contorno (Dirichlet) e o algoritmo de diferenças finitas para a atualização de células (stencil de 5 pontos). Serve como classe base para as implementações sequencial e paralela.

-   `heat_diffusion_sequential.py`:
    -   Implementa a solução sequencial da simulação de difusão de calor. Herdando de `heat_diffusion_base.py`, executa o cálculo em um único fluxo de controle, servindo como uma linha de base para comparação de desempenho.

-   `heat_diffusion_parallel.py`:
    -   Implementa a solução paralela utilizando o módulo `threading` do Python. Divide o trabalho de atualização da grade entre múltiplos threads, utilizando `threading.Barrier` para sincronização por iteração. Também herda de `heat_diffusion_base.py`.

-   `shared_utils.py`:
    -   Contém funções utilitárias e uma classe base para a implementação distribuída. Inclui funções de serialização/desserialização (`pickle` com prefixo de tamanho) para comunicação via sockets, e uma `BaseHeatDiffusion` que é utilizada pelas componentes distribuídas.

-   `heat_diffusion_master.py`:
    -   Implementa o componente Master da solução distribuída. Atua como orquestrador, dividindo a grade, distribuindo sub-grades e regiões de halo para os Workers, coletando resultados e coordenando as iterações via comunicação por sockets.

-   `heat_diffusion_worker.py`:
    -   Implementa o componente Worker da solução distribuída. Conecta-se ao Master, recebe sua sub-grade e halos, realiza os cálculos de difusão para sua porção e envia os resultados de volta ao Master via sockets.

-   `compare_solvers.py`:
    -   Um script utilitário para executar e comparar diretamente as implementações sequencial e paralela. Calcula e exibe métricas de desempenho como Speedup e Eficiência, e visualiza os resultados para verificar a correção e o comportamento da difusão.

## Como Executar

### Pré-requisitos

Certifique-se de ter Python 3.x e as seguintes bibliotecas instaladas:

```bash
pip install numpy matplotlib
