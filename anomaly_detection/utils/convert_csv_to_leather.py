import pandas as pd

path = '../../data/processed/cifar-10-100.csv'
df = pd.read_csv(path)
df.to_feather('../data.feather')
