import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_images():
    # plotting the first img
    df = pd.read_csv('../../data/processed/cifar-100.csv')
    pixels = df.iloc[0, 2:]
    image = pixels.values.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()
    #get 5 random images
    random_rows = df.sample(n=10)
    #pixel_columns = random_rows.iloc[,2:]

