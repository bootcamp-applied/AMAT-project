import pandas as pd
from visualization import Visualization

df = pd.read_csv('../DAL/cifar_10_100_db.csv')

X = df.drop(['label', 'is_train'], axis=1)
y = df['label']

visu = Visualization()
# visu.TSNE(X, y)
# visu.Pareto(y)
