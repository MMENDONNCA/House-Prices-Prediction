
# **House Prices: Análise Preditiva de Preços de Imóveis**

![Capa do Projeto](https://placehold.co/1200x400/0284C7/FFFFFF?text=House+Prices%3A+Advanced+Regression+Techniques&font=inter)

### [**Acesse o Notebook Completo com o Código**](https://colab.research.google.com/drive/1nEr6q3I0aF0Pntho3Mg9VIVSpiCzGHWJ?usp=sharing) | [**Veja a Competição no Kaggle**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## **1. Visão Geral do Projeto**

Este projeto de Data Science aborda o desafio de prever os preços de venda de imóveis na cidade de Ames, Iowa, utilizando um dataset com 79 variáveis descritivas. O objetivo principal foi construir um modelo de regressão robusto, aplicando um fluxo de trabalho completo que vai desde a **análise exploratória** e **limpeza de dados** até a **engenharia de features** e a **modelagem preditiva**.

O diferencial deste projeto reside na abordagem metódica para o tratamento de dados e na utilização de modelos de **regressão regularizada**, como o Lasso (L1), para gerir a alta dimensionalidade e selecionar automaticamente as features mais relevantes.

**Resultado:** O modelo final, utilizando Regressão Lasso, alcançou um **Public Score de 0.13393** na competição do Kaggle, posicionando-se de forma competitiva.

---

## **2. Metodologia e Insights**

A solução foi desenvolvida seguindo uma estrutura clara, focada em extrair o máximo de informação dos dados e construir um modelo generalizável.

### **2.1. Análise Exploratória de Dados (EDA) - Entendendo o Mercado**

A primeira etapa foi mergulhar nos dados para entender as suas características e descobrir padrões.

* **Insight 1: A Distribuição de Preços é Assimétrica.** O histograma da variável alvo, `SalePrice`, revelou uma forte assimetria positiva. Para normalizar esta distribuição e otimizar o desempenho dos modelos lineares, foi aplicada uma **transformação logarítmica (`np.log1p`)**. Esta simples transformação tornou a relação entre as features e o preço muito mais linear e estável.

* **Insight 2: A Qualidade Geral e a Área são Reis.** A análise de correlação mostrou que `OverallQual` (Qualidade Geral dos Materiais e Acabamento) e `GrLivArea` (Área de Estar Acima do Solo) são, de longe, as variáveis com o maior impacto positivo no preço. Isto confirma a intuição de mercado: qualidade e tamanho são os principais fatores de valorização.

* **Insight 3: A Localização Importa (e Muito).** Através de boxplots, foi possível visualizar uma grande variação de preços entre os diferentes bairros (`Neighborhood`). Bairros como `NridgHt` e `NoRidge` apresentaram medianas de preço significativamente mais altas, demonstrando que a localização é uma feature categórica de altíssimo valor preditivo.

### **2.2. Limpeza e Engenharia de Features - Transformando Dados em Valor**

Com os insights da EDA, a próxima fase foi preparar os dados para a modelagem.

* **Tratamento de Nulos Contextual:** Em vez de simplesmente eliminar dados, foi adotada uma estratégia contextual. Por exemplo, valores nulos em `GarageType` ou `PoolQC` foram preenchidos com a string `'None'`, transformando a ausência de uma característica numa informação útil para o modelo. Para `LotFrontage`, os valores nulos foram preenchidos com a mediana do bairro correspondente, uma abordagem muito mais precisa do que usar uma mediana global.

* **Criação de Features Relevantes:** Foram criadas novas variáveis para capturar informações que não estavam explícitas:
    * `HouseAge`: Idade da casa no momento da venda.
    * `YearsSinceRemod`: Anos desde a última remodelação.
    * `TotalSF`: Área total da casa, somando porão e andares.
    * Estas features sintéticas ajudaram o modelo a capturar melhor os efeitos de desvalorização e de investimentos em renovação.

### **2.3. Modelagem com Regressão Regularizada**

Para lidar com o grande número de features (mais de 250 após o *one-hot encoding*), a escolha recaiu sobre modelos de regressão regularizada.

* **Por que Lasso (L1)?** A Regressão Lasso não só previne o *overfitting* ao penalizar a complexidade do modelo, mas também realiza uma **seleção automática de features**, zerando os coeficientes das variáveis menos importantes. No contexto de mais de 250 features, esta capacidade é crucial para criar um modelo mais simples, interpretável e robusto.

* **Seleção do Modelo Final:** Foram avaliados 5 modelos distintos: Regressão Linear, Ridge (L2), Lasso (L1), Árvore de Decisão e KNN. A performance de cada um foi rigorosamente comparada através da métrica RMSE (Raiz do Erro Quadrático Médio) e da análise visual dos gráficos de dispersão. **O modelo Lasso destacou-se consistentemente, apresentando o menor RMSE e previsões mais estáveis.** A sua capacidade intrínseca de realizar seleção de features provou ser a mais eficaz para este dataset, resultando na sua escolha para a submissão final no Kaggle.

---

## **3. Ferramentas e Bibliotecas**

* **Linguagem:** `Python 3`
* **Análise e Manipulação:** `Pandas`, `NumPy`
* **Visualização:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-learn`

---

## **4. Conclusão e Próximos Passos**

Este projeto demonstra a aplicação prática de técnicas de regressão avançadas para resolver um problema de negócio real. O resultado obtido no Kaggle valida a eficácia da metodologia, especialmente a importância de uma análise exploratória detalhada e o uso de regularização para gerir a complexidade.

**Possíveis Melhorias Futuras:**

1.  **Otimização de Hiperparâmetros:** Utilizar `GridSearchCV` ou `RandomizedSearchCV` para encontrar o valor ótimo de `alpha` para os modelos Lasso e Ridge.
2.  **Modelos de Ensemble:** Experimentar com modelos mais poderosos como XGBoost, LightGBM ou CatBoost, que frequentemente lideram as tabelas em competições do Kaggle.
3.  **Stacking:** Combinar as previsões de múltiplos modelos (ex: uma média ponderada entre o Ridge e o XGBoost) para criar um "super modelo" ainda mais preciso.
