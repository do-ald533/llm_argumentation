import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

# Ler os arquivos Excel
df1 = pd.read_excel('evaluation1.xlsx')
df2 = pd.read_excel('evaluation2.xlsx')

# Garantir que os dois arquivos tenham as mesmas colunas
assert list(df1.columns) == list(df2.columns), "As colunas dos dois arquivos devem ser idênticas."

# Calcular média e desvio padrão para cada anotador
print("=== Estatísticas descritivas ===")
for name, df in zip(['Anotador 1', 'Anotador 2'], [df1, df2]):
    print(f"\n{name}")
    print(df.describe().loc[['mean', 'std']])

# Calcular médias e desvios combinados (entre os dois anotadores)
combined = pd.concat([df1, df2])
print("\n=== Média e desvio padrão combinados ===")
print(combined.describe().loc[['mean', 'std']])

# Calcular acordo entre anotadores
print("\n=== Acordo entre anotadores ===")
for col in df1.columns:
    notas1 = df1[col]
    notas2 = df2[col]
    corr, _ = pearsonr(notas1, notas2)
    kappa = cohen_kappa_score(notas1, notas2, weights='quadratic')
    print(f"{col}: correlação de Pearson = {corr:.2f}, kappa = {kappa:.2f}")