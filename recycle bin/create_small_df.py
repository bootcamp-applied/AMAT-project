import pandas as pd
import os

path_cifar_10 = '../data/processed/cifar_10.csv'
df_cifar_10 = pd.read_csv(path_cifar_10)

new_df = df_cifar_10[:2000]
path = '../data/processed/small_df.csv'

if os.path.exists(path):
    os.remove(path)

new_df.to_csv(path, index=False, encoding='utf-8', mode='w')

