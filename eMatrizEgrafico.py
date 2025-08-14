import os
import cv2
import numpy as np
import gudhi
import time
import matplotlib.pyplot as plt

pasta_imagens = r"C:\Users\Yasmin\OneDrive\Imagens\Documentos\complexos\frontal_images"
pasta_saidas = r"C:\Users\Yasmin\OneDrive\Imagens\Documentos\complexos\saidas"
pasta_graficos = os.path.join(pasta_saidas, "diagramas")


def criar_pastas(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def calcular_gargalo(diag1, diag2):

    return gudhi.bottleneck_distance(diag1, diag2)


def main():
    criar_pastas(pasta_saidas, pasta_graficos)
    inicio_total = time.time()


    processed_bases = set()

    #gera diagramas de persistência e salva matrizes
    for nome_arquivo in os.listdir(pasta_imagens):
        caminho_imagem = os.path.join(pasta_imagens, nome_arquivo)
        nome_base = os.path.splitext(nome_arquivo)[0]  # ex: '1a', '1b'

        # carrega imagem em tons de cinza
        img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erro ao ler imagem: {caminho_imagem}")
            continue

        img_invertida = 255 - img
        img_float = img_invertida.astype(float)

        # calculo do diagrama de persistência
        t0 = time.time()
        cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=img_float)
        diag = cubical_complex.persistence(
            homology_coeff_field=2,
            min_persistence=0
        )

        # extrai [dim, nascimento, morte]
        matriz = np.array(
            [[dim, nascim, morte]
             for dim, (nascim, morte) in diag
             if morte != float('inf')],
            dtype=float
        )

        duracao = time.time() - t0
        print(f"Processado: {nome_arquivo} em {duracao:.2f} segundos.")

        # salva matriz .npy
        caminho_npy = os.path.join(pasta_saidas, nome_base + ".npy")
        np.save(caminho_npy, matriz)
        processed_bases.add(nome_base)

       #grafico
        if matriz.size > 0:
            plt.figure(figsize=(6, 6))
            for dim in np.unique(matriz[:, 0]):
                sel = matriz[matriz[:, 0] == dim]
                plt.scatter(sel[:, 1], sel[:, 2], label=f"Dim {int(dim)}", s=20, alpha=0.6)
            max_val = np.max(matriz[:, 2])
            plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
            plt.xlabel("Nascimento")
            plt.ylabel("Morte")
            plt.title(f"Diagrama de Persistência - {nome_base}")
            plt.legend()
            plt.grid(True)

            caminho_png = os.path.join(pasta_graficos, nome_base + ".png")
            plt.savefig(caminho_png)
            plt.close()

   # cálculo da distância do gargalo entre todas as imagens
    print("\nComputando distâncias bottleneck entre todas as imagens...")
    resultados = []
    bases_ordenadas = sorted(processed_bases)

    for i in range(len(bases_ordenadas)):
     for j in range(i + 1, len(bases_ordenadas)):
         nome_i = bases_ordenadas[i]
         nome_j = bases_ordenadas[j]
         path_i = os.path.join(pasta_saidas, nome_i + ".npy")
         path_j = os.path.join(pasta_saidas, nome_j + ".npy")

         if os.path.exists(path_i) and os.path.exists(path_j):
             m_i = np.load(path_i)
             m_j = np.load(path_j)
             diag_i = [(row[1], row[2]) for row in m_i]
             diag_j = [(row[1], row[2]) for row in m_j]
             d = calcular_gargalo(diag_i, diag_j)
             resultados.append((nome_i, nome_j, d))
             print(f"{nome_i} vs {nome_j}: {d:.4f}")
         else:
             faltando = []
             if not os.path.exists(path_i): faltando.append(nome_i)
             if not os.path.exists(path_j): faltando.append(nome_j)
             print(f"Arquivos .npy faltando para comparação: {' '.join(faltando)}")


    #csv
    csv_path = os.path.join(pasta_saidas, "dist_Gargalo_Resul.csv")
    with open(csv_path, 'w') as f:
         f.write("imagem1,imagem2,dist_gargalo\n")
         for nome_i, nome_j, dist in resultados:
            f.write(f"{nome_i},{nome_j},{dist}\n")


    duracao_total = time.time() - inicio_total
    print(f"\nTempo total de processamento: {duracao_total:.2f} segundos.")

if __name__ == "__main__":
    main()
