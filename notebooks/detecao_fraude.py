#Importação de bibliotecas principais
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

#Para o XGBoost e o SHAP
from xgboost import XGBClassifier
import shap

#Carregando o dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "mlg-ulb/creditcardfraud",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
print("First 5 records:", df.head())

#Primeira inspeção
df.shape
df.info()
df.isna().sum().sum()
df["Class"].value_counts()
df["Class"].value_counts(normalize=True) * 100
#O objetivo é mostrar explicitamente a proporção das classes. Nesse conjunto, existem 492 fraudes em 284.807 registros, o que representa aproximadamente 0,17% dos dados.

#Por que a acurácia engana?
total = len(df)
fraudes = df["Class"].sum()
normais = total - fraudes

acuracia_modelo1= normais / total
acuracia_modelo1
#Um classificador que sempre responde 0 pode obter aproximadamente 99,8% de acurácia, mas terá:

#Recall da fraude: 0
#Precisçao de fraude : Indefinida ou 0
#F1 da fraude:0

#Isso acontece porque a acurácia mede o total de acertos, enquanto o problema está justamente em encontrar a classe rara.
#Recall: entre todas as fraudes reais, quantas foram encontradas.
#Precisão: entre todas as transações marcadas como fraude, quantas realmente eram fraude.
#F1: equilíbrio entre precisão e recall.
#ROC-AUC: capacidade de ordenar as classes em diferentes limiares.
#PR-AUC: geralmente mais informativa em problemas com classe positiva muito rara.

#Para fraude, use o recall da classe 1 como métrica principal, mas não ignore a precisão. Um recall alto pode gerar uma quantidade impraticável de falsos positivos.

#Preparação dos dados

#A coluna "Amount" tem uma distribuição altamente assimétrica, com a maioria das transações concentradas em valores baixos e algumas transações de valores muito altos. Para lidar com isso, aplicamos a transformação logarítmica para reduzir o impacto dos valores extremos e melhorar a performance do modelo.
df["LogAmount"] = np.log1p(df["Amount"])
#Retirando a coluna "Amount" do conjunto de features, pois agora temos a versão transformada "LogAmount". A coluna "Class" é a variável alvo e não deve ser incluída nas features.
features = [col for col in df.columns if col not in ["Class", "Amount"]]
features.append("LogAmount")

X = df[features]
y = df["Class"]

#Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#O stratify=y garante que a proporção de fraudes permaneça semelhante nos conjuntos de treino e teste.

#O escalonamento deve ser feito dentro de um Pipeline, evitando vazamento de dados:
scaler = StandardScaler()


#MODELOS

#Regressçao Logística
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

proba_logistic = logistic_model.predict_proba(X_test)[:, 1]
pred_logistic = (proba_logistic >= 0.5).astype(int)
#O parâmetro class_weight="balanced" aumenta a importância da classe minoritária durante o treinamento.

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

proba_rf = rf_model.predict_proba(X_test)[:, 1]
pred_rf = (proba_rf >= 0.5).astype(int)
#Como o Random Forest não depende da mesma forma da escala das variáveis, ele pode ser treinado sem StandardScaler.

#XGBoost
#Proporção entre classes:
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
scale_pos_weight
#Treinando o modelo
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
pred_xgb = (proba_xgb >= 0.5).astype(int)
#O eval_metric="aucpr" é coerente com um problema em que a classe positiva é rara.

#AVALIAÇÃO DOS MODELOS
def avaliar_modelo(nome, y_true, probabilidades, threshold=0.5):
    predicoes = (probabilidades >= threshold).astype(int)

    return {
        "modelo": nome,
        "limiar": threshold,
        "precision_fraud": precision_score(
            y_true, predicoes, zero_division=0
        ),
        "recall_fraud": recall_score(
            y_true, predicoes, zero_division=0
        ),
        "f1_fraud": f1_score(
            y_true, predicoes, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilidades),
        "pr_auc": average_precision_score(y_true, probabilidades)
    }
#Comparando Modelos
resultados = pd.DataFrame([
    avaliar_modelo("Regressão Logística", y_test, proba_logistic),
    avaliar_modelo("Random Forest", y_test, proba_rf),
    avaliar_modelo("XGBoost", y_test, proba_xgb)
])

resultados.sort_values("recall_fraud", ascending=False)
#relatório detalhado do modelo escolhido
print(classification_report(
    y_test,
    (proba_xgb >= 0.5).astype(int),
    target_names=["Normal", "Fraude"],
    zero_division=0
))
#Matriz de confusão
cm = confusion_matrix(
    y_test,
    (proba_xgb >= 0.5).astype(int)
)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de confusão")
plt.show()

#Ajuste do limiar de decisão
#O limiar padrão de 0,5 não precisa ser o melhor para fraude. Um limiar menor normalmente aumenta o recall, mas também pode aumentar os falsos positivos.
thresholds = np.arange(0.05, 0.96, 0.05)

threshold_results = []

for threshold in thresholds:
    predictions = (proba_xgb >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df

plt.figure(figsize=(10, 5))

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    marker="o",
    label="Precisão"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    marker="o",
    label="F1"
)

plt.xlabel("Limiar")
plt.ylabel("Métrica")
plt.title("Métricas por limiar de decisão")
plt.legend()
plt.grid()
plt.show()
#maximizar o F1
best_threshold = threshold_df.loc[
    threshold_df["f1"].idxmax(),
    "threshold"
]

best_threshold
