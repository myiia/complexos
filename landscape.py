import os
import numpy as np
import matplotlib.pyplot as plt

def lambda_func(b, d, x_vals):
    meio = (b + d) / 2 #ponto medio
    altura = (d - b) / 2 #altura maxima da funcao triangular
    return np.maximum(0, altura - np.abs(x_vals - meio)) #funcao triangular e valores negativos sao truncados pra 0

def gerar_landscap(saida_dir, pontos=1000, num_camadas=5):
    
     # qtd de pontos no dominio x onde as func. sao avaliadas
     #gerar as paisagens de persistência para dimensões 0 e 1 a partir dos arqv. .npy
    pasta_matrizes = os.path.join(saida_dir, "matrizesBarr")
    pasta_csv = os.path.join(saida_dir, "landscape_csv_manual")
    pasta_plot = os.path.join(saida_dir, "landscape_plot_manual")

    os.makedirs(pasta_csv, exist_ok=True)
    os.makedirs(pasta_plot, exist_ok=True)

    for arquivo in sorted(os.listdir(pasta_matrizes)):
        if not arquivo.endswith(".npy"):
            continue

        caminho = os.path.join(pasta_matrizes, arquivo)
        matriz = np.load(caminho)

        if matriz.size == 0:
            print(f"Matriz vazia: {arquivo}")
            continue

        base = os.path.splitext(arquivo)[0]
        print(f"Processando paisagem conjunta de {base}...")

        paisagens = {} #dicionario que armazena os dados dos intervalos de persist.
        min_nasc = np.inf #valor infinito positivo. vai armazenar o menor valor de nasc. entre todos os intervalos (define lim. inferior onde a paisagem sera avaliada)
        max_mort = -np.inf #valor infinito negativo. armazena o maior valor de morte entre todos os intervalos (define lim. superior onde a paisagem sera avaliada)

        #coleta de dados e definicao do dominio
        for dim in [0, 1]: #para cada dim (0 e 1) seleciona (b,d)
            dados = matriz[matriz[:, 0] == dim][:, 1:3]
            dados = dados[dados[:, 1] > dados[:, 0]]
            if len(dados) == 0: #se nao tiver nenhum (b, d) -> ignora
                continue
            paisagens[dim] = dados #armazena os pares validos
            min_nasc = min(min_nasc, dados[:, 0].min()) #atualiza o infimo do dominio
            max_mort = max(max_mort, dados[:, 1].max())

        if not paisagens: #verifica se nao foram encontrados dados validos
            continue

        x_vals = np.linspace(min_nasc, max_mort, pontos) #cria um vetor de pontos uniformemente espacados entre o menor nasc. e a maior mort

        # construcao das paisagens
        curvas_por_dim = {}

        for dim in paisagens:
            dados = paisagens[dim]
            funcoes = np.array([lambda_func(b, d, x_vals) for b, d in dados]) #uma matriz onde cada linha é uma função triangular lambda avaliada em todos os pontos x
            landscap_camadas = np.zeros((num_camadas, pontos))
            for i in range(pontos):
                top_k = -np.sort(-funcoes[:, i]) #ordena os valores lamda top_k representa os valores mais altos das funcoes no ponti x_i
                k = min(num_camadas, len(top_k)) #seleciona as k maiores para formar as camadas de paisagem
                landscap_camadas[:k, i] = top_k[:k] #preenche com as k maiores alturas
            curvas_por_dim[dim] = landscap_camadas

     

        #grafico
        plt.figure()
        cores = {0: "blue", 1: "red"}
        estilos = {0: "-", 1: "-"}
        for dim in curvas_por_dim:
            for k in range(num_camadas):
                plt.plot(x_vals, curvas_por_dim[dim][k],
                         linestyle=estilos[dim], color=cores[dim],
                         label=f"H{dim} λ{k+1}", alpha=0.7)
        plt.title(f"Paisagem de Persistência - {base}")
        plt.xlabel("Tempo")
        plt.ylabel("λ_k(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_plot, f"{base}_H0_H1_landscape.png"), dpi=300)
        plt.close()

        #csv
        colunas_csv = [x_vals]
        headers = ["x"]
        for dim in [0, 1]:
            if dim in curvas_por_dim:
                for k in range(num_camadas):
                    colunas_csv.append(curvas_por_dim[dim][k])
                    headers.append(f"H{dim}_lambda_{k+1}")

        matriz_csv = np.column_stack(colunas_csv)
        csv_path = os.path.join(pasta_csv, f"{base}_H0_H1_landscape.csv")
        np.savetxt(csv_path, matriz_csv, delimiter=",", header=",".join(headers), comments='')

        print(f"Paisagem conjunta gerada: {base}")
















if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python landscape.py <diretorio_saida>")
        sys.exit(1)
    saida_dir = sys.argv[1]
    gerar_landscap(saida_dir)
#python landscape.py C:/Users/Yasmin/OneDrive/Imagens/Documentos/complexos/saidas
