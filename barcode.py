import os
import cv2
import numpy as np
import gudhi
import matplotlib
import matplotlib.pyplot as plt
from gudhi import bottleneck_distance
from itertools import combinations
import csv

# Desativa o uso de LaTeX para evitar warnings
matplotlib.rcParams['text.usetex'] = False

def gerar_barcodes(frontal_dir, saida_dir):
    # Pastas de saída
    pasta_npy = os.path.join(saida_dir, "matrizesBarr")
    pasta_barcodes = os.path.join(saida_dir, "barcodes")
    os.makedirs(pasta_npy, exist_ok=True)
    os.makedirs(pasta_barcodes, exist_ok=True)

    for arquivo in os.listdir(frontal_dir):
        caminho_imagem = os.path.join(frontal_dir, arquivo)
        # Carrega imagem
        img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Falha ao ler: {arquivo}")
            continue

        # Inversão e conversão para float
        img_invertida = 255 - img
        img_float = img_invertida.astype(float)

        # Cubical complex e persistência
        cc = gudhi.CubicalComplex(top_dimensional_cells=img_float)
        intervalos = cc.persistence(homology_coeff_field=2, min_persistence=0)

        # Salvar .npy com (dim, nascimento, morte)
        mat = [[d, b, c] for d, (b, c) in intervalos if c != float('inf')]
        mat = np.array(mat)
        base = os.path.splitext(arquivo)[0]
        npy_path = os.path.join(pasta_npy, f"{base}.npy")
        np.save(npy_path, mat)

        # Geração do código de barras no estilo tradicional
        plt.figure(figsize=(6, 6))
        colors = {0: 'red', 1: 'blue'}
        legendas = set()

        # Separar por dimensão
        dim_0 = [pair for pair in intervalos if pair[0] == 0 and pair[1][1] != float('inf')]
        dim_1 = [pair for pair in intervalos if pair[0] == 1 and pair[1][1] != float('inf')]

        todos = dim_0 + dim_1  # ordem vertical

        for idx, (dim, (birth, death)) in enumerate(todos):
            cor = colors.get(dim, 'gray')
            label = f"{dim}"
            if label not in legendas:
                plt.plot([birth, death], [idx, idx], color=cor, label=label)
                legendas.add(label)
            else:
                plt.plot([birth, death], [idx, idx], color=cor)

        plt.xlabel("Filtro (nível)")
        plt.ylabel("Intervalos")
        plt.title(f"codigo de barras - {base}")
        plt.legend(title="Dimension")
        plt.tight_layout()
        plt.gca().invert_yaxis()  # opcional: barra maior no topo

        out_png = os.path.join(pasta_barcodes, f"{base}_barcode.png")
        plt.savefig(out_png, dpi=300)
        plt.close()

        print(f"fim: {base}")

    # Após gerar todos os barcodes, calcular as distâncias
    calcular_distancias_bottleneck(pasta_npy, saida_dir)
