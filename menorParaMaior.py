import pandas as pd

# Carregar o arquivo (ajuste o separador se necessário)
df = pd.read_csv('dist_Gargalo_Result.csv', sep='|', skipinitialspace=True)

# Ordenar pela coluna 'imagem3' e resetar o índice
df_sorted = df.sort_values(by='imagem3').reset_index(drop=True)

# Salvar ou exibir
print(df_sorted)
df_sorted.to_csv('dados_ordenados.csv', index=False, sep='|')