import pandas as pd

path = '../../data/processed/cifar_10_100.csv'
df = pd.read_csv(path)
df.to_feather('../df.feather')
