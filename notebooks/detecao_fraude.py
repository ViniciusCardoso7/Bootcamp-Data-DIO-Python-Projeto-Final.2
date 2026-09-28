from XGBoost import XGBClassifier
import shap  #df is your dataframe
#frac is percentage of data to sample
#0.2 is 20 percent
df.sample(frac = 0.2)

url = 'https://raw.githubusercontent.com/selva86/datasets/master/CreditCard.csv'
df = pd.read_csv(url)

