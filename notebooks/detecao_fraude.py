import pandas as pd

url = 'https://https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data?select=creditcard.csv'

df = pd.read_csv(url)

df.head()
