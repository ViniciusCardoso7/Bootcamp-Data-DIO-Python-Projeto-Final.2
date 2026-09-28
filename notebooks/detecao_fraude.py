from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Baixa o dataset do Kaggle
pasta_dataset = Path(
    kagglehub.dataset_download("mlg-ulb/creditcardfraud")
)

# Localiza e carrega o arquivo CSV
arquivos_csv = list(pasta_dataset.rglob("*.csv"))
if not arquivos_csv:
    raise FileNotFoundError("Nenhum arquivo CSV foi encontrado no dataset.")

df = pd.read_csv(arquivos_csv[0])

print("Arquivo carregado:", arquivos_csv[0])
print("Dimensões:", df.shape)
print(df.head())

# Visualiza a quantidade de transações legítimas e fraudulentas
sns.countplot(data=df, x="Class")
plt.title("Transações legítimas (0) e fraudulentas (1)")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()