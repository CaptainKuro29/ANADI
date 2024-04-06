import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv("CO_data.csv") # ler data do ficheiro

# Limpeza de data: Lidar com valores NaN
data.fillna(0, inplace=True)

#Pergunta 4.1 (1

# Filtrar os data de Portugal
data_portugal = data[data['country'] == 'Portugal']

plt.figure(figsize=(10, 6))
plt.plot(data_portugal['year'], data_portugal['co2'])
plt.title('Emissões Totais de CO2 para Portugal (1900-2021)')
plt.xlabel('Ano')
plt.ylabel('Emissões de CO2')
plt.grid(True)
plt.show()

ano_max_co2 = data_portugal.loc[data_portugal['co2'].idxmax(), 'year']
max_co2 = data_portugal['co2'].max()
print(f"O ano com o máximo de emissões de CO2 para Portugal foi em {ano_max_co2} com {max_co2} milhões de toneladas.")

#Pergunta 4.1 (2

#Escolher as fontes
fontes = ['cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'methane', 'nitrous_oxide', 'oil_co2']

# Adicionar Labels ao gráfico
nomes_descritivos = {
    'cement_co2': 'Cimento',
    'coal_co2': 'Carvão',
    'flaring_co2': 'Queima',
    'gas_co2': 'Gás',
    'methane': 'Metano',
    'nitrous_oxide': 'Óxido Nitroso',
    'oil_co2': 'Petróleo'
}


plt.figure(figsize=(10, 6))

#Adicionar ao grafico uma linha para cada fonte de emissão
for fonte in fontes:
    plt.plot(data_portugal['year'], data_portugal[fonte], label=nomes_descritivos[fonte])

plt.title('Emissões de CO2 de Diferentes Fontes para Portugal (1900-2021)')
plt.xlabel('Ano')
plt.ylabel('Emissões de CO2')
plt.legend()
plt.grid(True)
plt.show()

#Pergunta 4.1 (3

data_espanha = data[data['country'] == 'Spain']

plt.figure(figsize=(10, 6))
plt.plot(data_portugal['year'], data_portugal['co2'] / data_portugal['population'], label='Portugal')
plt.plot(data_espanha['year'], data_espanha['co2'] / data_espanha['population'], label='Espanha')
plt.title('Emissões de CO2 per Capita para Portugal e Espanha (1900-2021)')
plt.xlabel('Ano')
plt.ylabel('Emissões de CO2 per Capita')
plt.legend()
plt.grid(True)
plt.show()

#Pergunta 4.1 (4

#Escolher os países
paises = ['United States', 'China', 'India', 'European Union (27)', 'Russian Federation']

#Recolher os data de carvão para cada país selecionado
data_carvao = data[data['country'].isin(paises) & (data['year'] >= 2000) & (data['year'] <= 2021)]

plt.figure(figsize=(10, 6))

#Adicionar uma linha para o valor de CO2 cada país 
for pais in paises:
    data_pais = data_carvao[data_carvao['country'] == pais]
    plt.plot(data_pais['year'], data_pais['coal_co2'], label=pais)
plt.title('Emissões de CO2 Originadas pelo Carvão (2000-2021)')
plt.xlabel('Ano')
plt.ylabel('Emissões de CO2')
plt.legend()
plt.grid(True)
plt.show()

#Pergunta 4.1 (5

# Selecionar os data relevantes para cada região
data_regioes = data[data['country'].isin(paises) & (data['year'] >= 2000) & (data['year'] <= 2021)]

# Calcular as médias
medias = data_regioes.groupby('country').mean()

# Formatar as médias para terem apenas três casas decimais
medias_formatadas = medias.round(3)

# Criar a tabela
tabela = medias_formatadas[['cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'methane', 'nitrous_oxide', 'oil_co2']]
tabela.columns = ['Cimento', 'Carvão', 'Queima', 'Gás', 'Metano', 'Óxido Nitroso', 'Petróleo']
tabela.index.name = 'Região'

# Mostrar a tabela
print(tabela)

#Pergunta 4.2 (1

# Definir a amostra aleatória de anos
seed_value = 100
years = pd.Series([i for i in range(1900, 2021)])
sampleyears1 = years.sample(n=30, replace=False, random_state=seed_value)

# Selecionar apenas os anos da amostra para Portugal
sample_years1_data_portugal = data[(data['country'] == 'Portugal') & (data['year'].isin(sampleyears1))]

# Calcular a média do PIB para Portugal na amostra
mean_gdp_sample_years1_portugal = sample_years1_data_portugal['gdp'].mean()

# Selecionar apenas os anos da amostra para Hungria
sample_years1_data_hungary = data[(data['country'] == 'Hungary') & (data['year'].isin(sampleyears1))]

# Calcular a média do PIB para Hungria na amostra
mean_gdp_sample_years1_hungary = sample_years1_data_hungary['gdp'].mean()

# Realizar o teste de hipótese
alpha = 0.05
t_statistic, p_value = stats.ttest_ind(sample_years1_data_portugal['gdp'], sample_years1_data_hungary['gdp'], alternative='greater')

if p_value < alpha:
    print("Rejeitamos a hipótese nula. A média do PIB de Portugal na amostra é estatisticamente superior à média do PIB da Hungria na amostra.")
else:
    print("Não rejeitamos a hipótese nula. Não há evidências suficientes para concluir que a média do PIB de Portugal na amostra é estatisticamente superior à média do PIB da Hungria na amostra.")

#Pergunta 4.2 (2

# Definir as amostras aleatórias de anos para Portugal e Hungria
seed_value = 55
years = pd.Series([i for i in range(1900, 2021)])
np.random.seed(seed_value)
sampleyears2 = years.sample(n=12, replace=False)

seed_value = 85
np.random.seed(seed_value)
sampleyears3 = years.sample(n=12, replace=False)

# Selecionar apenas os anos da amostra para Portugal
sample_years2_data_portugal = data[(data['country'] == 'Portugal') & (data['year'].isin(sampleyears2))]

# Calcular a média do PIB para Portugal na amostra
mean_gdp_sample_years2_portugal = sample_years2_data_portugal['gdp'].mean()

# Selecionar apenas os anos da amostra para Hungria
sample_years3_data_hungary = data[(data['country'] == 'Hungary') & (data['year'].isin(sampleyears3))]

# Calcular a média do PIB para Hungria na amostra
mean_gdp_sample_years3_hungary = sample_years3_data_hungary['gdp'].mean()

# Realizar o teste de hipótese
alpha = 0.05
t_statistic, p_value = stats.ttest_ind(sample_years2_data_portugal['gdp'], sample_years3_data_hungary['gdp'], alternative='greater')

if p_value < alpha:
    print("Rejeitamos a hipótese nula. A média do PIB de Portugal na amostra sampleyears2 é estatisticamente superior à média do PIB da Hungria na amostra sampleyears3.")
else:
    print("Não rejeitamos a hipótese nula. Não há evidências suficientes para concluir que a média do PIB de Portugal na amostra sampleyears2 é estatisticamente superior à média do PIB da Hungria na amostra sampleyears3.")

#Pergunta 4.2 (3

# Selecionar apenas os anos da amostra para os países relevantes
sample_years2_countries = ['United States', 'Russia', 'China', 'India', 'EU27']
sample_years2_data = data[(data['year'].isin(sampleyears2)) & (data['country'].isin(sample_years2_countries))]

# Realizar a ANOVA
modelo_anova = ols('co2 ~ country', data=sample_years2_data).fit()
anova_tabela = sm.stats.anova_lm(modelo_anova, typ=2)

print("Resultados da ANOVA:")
print(anova_tabela)

# Realizar o teste de Tukey para a análise post-hoc
compara_tukey = pairwise_tukeyhsd(endog=sample_years2_data['co2'], groups=sample_years2_data['country'], alpha=0.05)

print("\nComparação de médias (Tukey HSD):")
print(compara_tukey)

#Pergunta 4.3 (1

# Selecionar os data relevantes
regioes = ['Africa', 'Asia', 'South America', 'North America', 'Europe', 'Oceania']
data_carvao = data[(data['year'] >= 2000) & (data['year'] <= 2021) & (data['country'].isin(regioes))]

# Filtrar os data para incluir apenas as emissões de CO2 provenientes do carvão
data_carvao = data_carvao.pivot_table(index='year', columns='country', values='coal_co2')

# Calcular a tabela de correlação
tabela_correlacao = data_carvao.corr()

# Mostrar a tabela de correlação
print("Tabela de Correlação entre as Regiões:")
print(tabela_correlacao)
plt.figure(figsize=(8, 6)) 
sns.heatmap(tabela_correlacao,annot=True,  cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz das Correlações CO2-Carvão (2000-2021)')
plt.show()

#Pergunta 4.3 (2 

#a) 

data_filtrados = data[(data['year'] > 2000) & (data['year'] < 2022) & (data['year'] % 2 == 0)]
countrys = ['Germany', 'Russia', 'France', 'Portugal', 'Europe']
X = data_filtrados.pivot(index='year', columns='country', values='coal_co2')[countrys] 
X

X_value = X[['Germany', 'Russia', 'France', 'Portugal']]
Y = X['Europe'] 
X_pivot = sm.add_constant(X_value) 
model = sm.OLS(Y, X_value).fit() 
print(model.summary())

#b)

residuos = model.resid 
teste_shapiro = stats.shapiro(residuos)
print(f"Shapiro-Wilk: Statistics={teste_shapiro.statistic}, P-value={teste_shapiro.pvalue}")  
fig = sm.qqplot(residuos, line='s',markersize=5, alpha=0.5) 
plt.title('QQ') 
plt.show()  
plt.scatter(model.fittedvalues, residuos, alpha=0.5) 
plt.xlabel('Values') 
plt.ylabel('Residuos') 
plt.title('Homoscedasticidade')
plt.axhline(y=0, color='r', linestyle='--') 
plt.show() 
durbinWatson = durbin_watson(residuos)
print(' DW:', durbinWatson)

#c)

# Calcular VIF para cada variável independente
vif_data = X_pivot.assign(const=1)
vif_series = pd.Series([variance_inflation_factor(vif_data.values, i) 
                        for i in range(vif_data.shape[1])], 
                        index=vif_data.columns)

# Exibir os resultados
print("Fator de Inflação da Variância (VIF):\n", vif_series)

#d) 
# Os países têm VIFs moderados , o que significa que elas estão correlacionadas entre si, mas não de forma excessiva.

#e)

# Definir a lista de países
paises = ['Germany', 'Russia', 'France', 'Portugal']

# Filtrar os data para o ano de 2015
co2_2015 = data[data['year'] == 2015]

# Filtrar os data para os países específicos e para a Europa
co2_europa_2015 = co2_2015[co2_2015['country'].isin(paises)]

# Calcular a soma das emissões de CO2 do carvão para cada país e para a Europa em 2015
X_2015 = co2_europa_2015.pivot_table(index='year', columns='country', values='coal_co2').dropna()
Y_2015 = X_2015.sum(axis=1)

# Adicionar a constante ao modelo
X_2015_const = sm.add_constant(X_2015)

# Criar modelo de regressão linear
model = sm.OLS(Y_2015, X_2015_const).fit()

# Estimar a emissão de CO2 proveniente do carvão na Europa em 2015
Y_pred_2015 = model.predict(X_2015_const)

# Comparar com o valor real
print("\nEstimativa da Emissão de CO2 proveniente do carvão na Europa em 2015:")
print(Y_pred_2015)
print("\nValor Real da Emissão de CO2 proveniente do carvão na Europa em 2015:")
print(Y_2015)
