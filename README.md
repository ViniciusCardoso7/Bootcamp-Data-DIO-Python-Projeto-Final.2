# Bootcamp-Data-DIO-Python-Projeto-Final.2/Detecção de Fraude em Cartões
Projeto de classificação supervisionada para identificar transações fraudulentas em cartões de crédito usando Python e técnicas de machine learning.

O projeto percorre as principais etapas de um pipeline de dados:

- Exploração da base;
- Tratamento do desbalanceamento;
- Preparação das variáveis;
- Treinamento de modelos;
- Avaliação por métricas adequadas;
- Ajuste do limiar de decisão;
- Explicação das previsões.

## Objetivo

O objetivo é construir um modelo capaz de identificar transações fraudulentas, priorizando o reconhecimento da classe `Fraude`.

Este é um problema de classificação desbalanceada: a maior parte das transações é legítima, enquanto uma pequena parcela representa fraudes. Por isso, a acurácia não deve ser usada como principal critério de avaliação.

Um modelo que classificasse todas as transações como normais poderia apresentar uma acurácia muito alta, mas não encontraria nenhuma fraude. Neste projeto, as principais métricas analisadas são:

- Precisão da classe fraude;
- Recall da classe fraude;
- F1-score da classe fraude;
- ROC-AUC;
- PR-AUC.

## Dataset

A base utilizada é o dataset público **Credit Card Fraud Detection**, disponibilizado no Kaggle:

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

O dataset contém as seguintes informações principais:

- `Time`: tempo decorrido entre a transação e a primeira transação;
- `Amount`: valor da transação;
- `V1` a `V28`: variáveis transformadas por PCA;
- `Class`: variável-alvo, em que `0` representa uma transação normal e `1` representa uma fraude.

Por questões de privacidade, as variáveis originais foram anonimizadas e transformadas por PCA.

O arquivo de dados não é armazenado neste repositório. O notebook carrega a base a partir do link ou segue as instruções documentadas para obtenção do dataset.

## Desbalanceamento

A distribuição das classes observada no notebook foi:

| Classe | Significado | Percentual |
|---|---|---:|
| 0 | Transação normal | 99,827251% |
| 1 | Fraude | 0,172749% |

Esse desbalanceamento faz com que a acurácia seja uma métrica inadequada para representar sozinha a qualidade do modelo.

Neste projeto, o recall da classe fraude recebe atenção especial porque mede a proporção de fraudes reais que foram identificadas pelo modelo.

## Preparação dos dados

As etapas de preparação realizadas foram:

1. Carregamento da base com `pandas`.
2. Inspeção das colunas e dos tipos de dados.
3. Verificação de valores ausentes.
4. Análise da distribuição da variável `Class`.
5. Criação da variável `LogAmount` a partir de `Amount`.
6. Separação entre variáveis preditoras e variável-alvo.
7. Divisão dos dados em treino e teste com `stratify`.
8. Padronização das variáveis numéricas com `StandardScaler`.
9. Aplicação de técnicas de balanceamento somente no conjunto de treinamento.

A transformação logarítmica foi utilizada para reduzir a influência de valores muito altos na variável de valor da transação:

```python
df["LogAmount"] = np.log1p(df["Amount"])
```

A divisão estratificada foi usada para preservar a proporção de fraudes nos conjuntos de treino e teste.

## Técnicas de balanceamento

Foram testadas diferentes estratégias para lidar com o desbalanceamento:

- SMOTE;
- NearMiss;
- Random oversampling;
- Random undersampling.

O objetivo foi verificar como cada técnica afetava o recall, a precisão e o F1-score da classe fraude.

O balanceamento foi aplicado somente aos dados de treinamento. O conjunto de teste permaneceu com sua distribuição original para representar melhor o comportamento esperado em dados reais.

## Modelos

Os modelos avaliados foram:

- Regressão logística como baseline;
- Random Forest;
- XGBoost.

Os modelos foram comparados utilizando as probabilidades previstas e diferentes métricas para a classe positiva, representada pelas fraudes.

## Resultados

Os resultados abaixo foram obtidos nos testes de balanceamento registrados no notebook:

| Estratégia | Limiar | Precisão fraude | Recall fraude | F1 fraude | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SMOTE | 0,50 | 0,058102 | 0,918367 | 0,109290 | 0,970966 | 0,721528 |
| NearMiss | 0,50 | 0,004019 | 0,959184 | 0,008005 | 0,942822 | 0,208816 |
| Oversampling | 0,50 | 0,060484 | 0,918367 | 0,113493 | 0,971050 | 0,711185 |
| Undersampling | 0,50 | 0,037283 | 0,918367 | 0,071656 | 0,976577 | 0,606680 |

Os resultados mostram que o NearMiss apresentou o maior recall entre as estratégias avaliadas, mas sua precisão e seu F1-score foram muito baixos. Isso indica que o modelo marcou muitas transações normais como fraudulentas.

O oversampling apresentou o maior F1-score entre os resultados exibidos na tabela, além de um recall elevado. Ainda assim, sua precisão permaneceu baixa, o que demonstra a existência de muitos falsos positivos.

A escolha final do modelo não deve considerar somente o maior recall. É necessário avaliar o custo operacional de investigar transações marcadas incorretamente como fraude.

## Interpretação das métricas

### Recall

O recall responde:

> Entre todas as fraudes que realmente ocorreram, quantas foram identificadas?

Um recall alto reduz a quantidade de fraudes não detectadas.

### Precisão

A precisão responde:

> Entre todas as transações classificadas como fraude, quantas realmente eram fraude?

Uma precisão baixa significa que muitas transações legítimas foram sinalizadas incorretamente.

### F1-score

O F1-score combina precisão e recall em uma única métrica. Ele é útil quando é necessário equilibrar a detecção de fraudes e o controle de falsos positivos.

## Limiar de decisão

O modelo normalmente classifica uma transação como fraude quando sua probabilidade prevista é maior ou igual a `0,50`.

Entretanto, esse limiar pode ser ajustado de acordo com o objetivo do negócio.

- Um limiar menor tende a aumentar o recall.
- Um limiar maior tende a aumentar a precisão.
- O melhor limiar depende do custo de deixar uma fraude passar e do custo de investigar uma transação legítima.

Neste projeto, foram avaliados diferentes limiares utilizando:

- Precisão;
- Recall;
- F1-score.

O limiar final deve ser escolhido com base em uma regra explícita, como:

- Maximizar o F1-score;
- Garantir recall mínimo de 80%;
- Maximizar a precisão mantendo um recall aceitável.


## Curvas de avaliação

O notebook apresenta:

- Matriz de confusão;
- Curva ROC;
- Curva de precisão e recall;
- Comparação das métricas por limiar.

A curva de precisão e recall é especialmente importante neste projeto porque a classe fraude é muito rara.

## Explicabilidade com SHAP

O SHAP foi utilizado para analisar como as variáveis contribuíram para as previsões do modelo.

A análise busca responder:

- Quais variáveis tiveram maior impacto global?
- Quais variáveis aumentaram a probabilidade prevista de fraude?
- Quais variáveis reduziram essa probabilidade?
- Como explicar uma previsão individual?

As variáveis `V1` a `V28` são componentes anonimizados obtidos por PCA. Portanto, é possível interpretar a direção e a intensidade da contribuição de uma variável para uma previsão, mas não atribuir diretamente um significado de negócio específico a cada componente.


## O que foi desenvolvido além do exemplo

Em relação ao fluxo inicial apresentado na aula, este projeto inclui:

- Criação da variável `LogAmount`;
- Avaliação de múltiplas técnicas de balanceamento;
- Comparação entre precisão, recall e F1-score;
- Uso de ROC-AUC e PR-AUC;
- Análise do efeito do limiar de decisão;
- Visualização das curvas ROC e precisão-recall;
- Interpretação das previsões com SHAP;
- Comparação dos impactos dos falsos positivos e falsos negativos.

## Tecnologias utilizadas

- Python;
- NumPy;
- Pandas;
- Matplotlib;
- Seaborn;
- Scikit-learn;
- XGBoost;
- SHAP;
- Imbalanced-learn;
- Jupyter Notebook.

## Como executar

Clone o repositório:

```bash
git clone https://github.com/ViniciusCardoso7/Bootcamp-Data-DIO-Python-Projeto-Final.2.git
cd Bootcamp-Data-DIO-Python-Projeto-Final.2
```

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No macOS ou Linux:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Inicie o Jupyter Notebook:

```bash
jupyter notebook
```

Depois, abra o notebook localizado em:

```text
notebooks/deteccao_fraude.ipynb
```

Execute as células na ordem, do início ao fim.

## Estrutura do projeto

```text
credit-card-fraud-detection/
├── notebooks/
│   └── deteccao_fraude.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

O dataset não deve ser enviado ao repositório.

## Limitações

Este projeto possui algumas limitações:

- As variáveis PCA são anonimizadas;
- O dataset representa um contexto específico de transações;
- Os resultados podem mudar de acordo com a divisão dos dados e os hiperparâmetros;
- O modelo não foi colocado em produção;
- O limiar escolhido não representa necessariamente uma política definitiva de aprovação ou bloqueio;
- A análise não inclui informações sobre o custo financeiro de cada tipo de erro.

## Próximos passos

Como melhorias futuras, podem ser realizados:

- Ajuste de hiperparâmetros com `GridSearchCV` ou `RandomizedSearchCV`;
- Validação cruzada estratificada;
- Teste de diferentes limiares em uma análise de custo;
- Criação de variáveis comportamentais ao longo do tempo;
- Comparação com outros modelos;
- Monitoramento de alteração na distribuição dos dados;
- Desenvolvimento de uma API para fazer previsões;
- Criação de um dashboard de acompanhamento das transações.

## Autor

Desenvolvido por **Vinicius Cardoso** como projeto de estudo em análise de dados e machine learning.

[LinkedIn](https://www.linkedin.com/in/viniciuscardoso2020/) | [GitHub](https://github.com/ViniciusCardoso7)
