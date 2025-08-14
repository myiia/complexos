import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import mode

# === CONFIGURAÇÃO DE CAMINHOS ===
pasta_landscapes = r"C:\Users\Yasmin\OneDrive\Imagens\Documentos\complexos\saidas\landscape_csv_manual"
pasta_resultados = os.path.join(pasta_landscapes, 'resultados_agrupamento')
os.makedirs(pasta_resultados, exist_ok=True)

# === CARREGAMENTO DOS DADOS ===
arquivos = sorted([f for f in os.listdir(pasta_landscapes) if f.endswith('.csv')])
dados = []
nomes_arquivos = []

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta_landscapes, nome_arquivo)
    df = pd.read_csv(caminho)
    vetor = df.values.flatten()
    dados.append(vetor)
    nomes_arquivos.append(nome_arquivo)

# === PADRONIZAÇÃO DOS VETORES (PADDING) ===
tamanho_max = max(len(v) for v in dados)
dados_padronizados = np.array([np.pad(v, (0, tamanho_max - len(v)), mode='constant') for v in dados])
X = np.array(dados_padronizados)

# === NORMALIZAÇÃO ===
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# === ESCOLHA DO MELHOR k COM SILHOUETTE ===
scores = []
ks = range(2, 26)
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=0)
    rotulos = kmeans.fit_predict(X_normalizado)
    score = silhouette_score(X_normalizado, rotulos)
    scores.append(score)
    print(f"k = {k} → Silhouette = {score:.4f}")

# Salva curva da Silhouette
df_scores = pd.DataFrame({'k': list(ks), 'silhouette_score': scores})
df_scores.to_csv(os.path.join(pasta_resultados, 'silhouette_scores.csv'), index=False)

plt.figure(figsize=(8, 5))
plt.plot(ks, scores, marker='o')
plt.title("Curva da Silhouette para diferentes valores de k (Landscapes)")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Score da Silhouette")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pasta_resultados, 'silhouette_plot.png'))
plt.show()

# === APLICAÇÃO DO KMeans COM O MELHOR k ===
melhor_k = ks[np.argmax(scores)]
print(f"\nMelhor número de clusters: k = {melhor_k}")

kmeans_final = KMeans(n_clusters=melhor_k, random_state=0)
rotulos_finais = kmeans_final.fit_predict(X_normalizado)

# Salva os rótulos dos clusters
df_rotulos = pd.DataFrame({'arquivo': nomes_arquivos, 'cluster': rotulos_finais})
df_rotulos.to_csv(os.path.join(pasta_resultados, 'labels_clusters.csv'), index=False)

# === RÓTULOS REAIS ===
rotulos_reais_dict = {}
for nome in nomes_arquivos:
    prefixo = nome.lower().split('_')[0]
    if prefixo.endswith('a'):
        rotulos_reais_dict[nome] = 'classeA'
    elif prefixo.endswith('b'):
        rotulos_reais_dict[nome] = 'classeB'
    else:
        raise ValueError(f"Formato desconhecido no nome do arquivo: {nome}")

rotulos_reais = [rotulos_reais_dict[nome] for nome in nomes_arquivos]
le = LabelEncoder()
rotulos_reais_int = le.fit_transform(rotulos_reais)

# === REALINHAMENTO DOS CLUSTERS COM OS RÓTULOS REAIS ===
def realinhar_clusters(rotulos_pred, rotulos_reais):
    rotulos_alinhados = np.zeros_like(rotulos_pred)
    for cluster in np.unique(rotulos_pred):
        mask = rotulos_pred == cluster
        rotulos_alinhados[mask] = mode(rotulos_reais[mask], keepdims=True)[0]
    return rotulos_alinhados

rotulos_finais_alinhados = realinhar_clusters(rotulos_finais, rotulos_reais_int)

# === MATRIZ DE CONFUSÃO ALINHADA ===
matriz = confusion_matrix(rotulos_reais_int, rotulos_finais_alinhados)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão Alinhada - Persistence Landscapes')
plt.tight_layout()
plt.savefig(os.path.join(pasta_resultados, 'matriz_confusao_alinhada.png'))
plt.show()



# === VISUALIZAÇÃO COM PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalizado)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rotulos_finais_alinhados, cmap='tab20', s=60)
plt.title(f"Clusters das Persistence Landscapes (k = {melhor_k})")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pasta_resultados, 'clusters_pca.png'))
plt.show()
