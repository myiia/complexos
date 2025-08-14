import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Caminho da imagem específica (altere para o nome correto)
nome_imagem = '72b.jpg'  # exemplo
pasta = r'C:\Users\Yasmin\OneDrive\Imagens\Documentos\complexos\frontal_images'
caminho = os.path.join(pasta, nome_imagem)

# Leitura da imagem em tons de cinza
img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Erro ao carregar imagem: {caminho}")
else:
    # Inversão e conversão para float
    img_invertida = 255 - img
    img_float = img_invertida.astype(float)

    # Plotagem lado a lado
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original (Grayscale)')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Invertida (255 - img)')
    plt.imshow(img_invertida, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Float (visualmente igual)')
    plt.imshow(img_float, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

