import os
import numpy as np
import matplotlib.pyplot as plt

def gerar_curvas_betti(saida_dir):
    pasta_matrizes = os.path.join(saida_dir, "matrizesBarr")
    pasta_csv = os.path.join(saida_dir, "curvasBetti_csv")
    pasta_plot = os.path.join(saida_dir, "curvasBetti_plot")
    os.makedirs(pasta_csv, exist_ok=True)
    os.makedirs(pasta_plot, exist_ok=True)
    for arquivo in os.listdir(pasta_matrizes):
        if not arquivo.endswith(".npy"):
            continue
        caminho = os.path.join(pasta_matrizes, arquivo)
        matriz = np.load(caminho)
        if matriz.size == 0:
            print(f"Matriz vazia: {arquivo}")
            continue
        base = os.path.splitext(arquivo)[0]

        
        #dominio t para qual as curvas serao computadas
        max_time = int(np.ceil(matriz[:, 2].max()))
        tempo = np.arange(0, max_time + 1)

        curvas = {0: np.zeros_like(tempo), 1: np.zeros_like(tempo)}
         #calculo de b0 e b1
        for dim in [0, 1]:
            #para cada instante t verifica quais [b, d) satisfazem: b1<=t<d (quais classes estao vivas no instante)
            dados = matriz[matriz[:, 0] == dim] #filtra os intervalos da dimensao que esta sendo vista (0 ou 1)
            for idx, t in enumerate(tempo):
                ativos = np.logical_and(dados[:, 1] <= t, dados[:, 2] > t) #verifica os ciclos vivos em t
                curvas[dim][idx] = np.count_nonzero(ativos) #qtd de ciclos vivos

        
        plt.figure()
        plt.plot(tempo, curvas[0], label="H0", color="blue")
        plt.plot(tempo, curvas[1], label="H1", color="orange")
        plt.xlabel("Tempo")
        plt.ylabel("Número de classes (β)")
        plt.title(f"Curva de Betti - {base}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(pasta_plot, f"{base}_H0_H1.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    
        csv_path = os.path.join(pasta_csv, f"{base}_H0_H1.csv")
        matriz_saida = np.column_stack((tempo, curvas[0], curvas[1]))
        np.savetxt(csv_path, matriz_saida, delimiter=",", header="tempo,betti_H0,betti_H1", comments='')
        print(f"Curva de Betti gerada: {base}")










if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python curvasBetti.py <diretório_de_saída>")
        sys.exit(1)

    saida_dir = sys.argv[1]
    gerar_curvas_betti(saida_dir)
