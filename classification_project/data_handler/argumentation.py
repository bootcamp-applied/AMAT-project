import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = './pictures/before_rotate/ship_3.csv'
df_img = pd.read_csv(path)
label = df_img['label']
pixels = [col for col in df_img.columns if col.startswith('pixel')]
df_img = df_img[pixels]