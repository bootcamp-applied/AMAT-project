import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_images(images):
    for image in images:
    plt.imshow(image)
    plt.show()

def load_new_images():
    df = pd.read_csv('../../data/processed/cifar-100.csv')
    #random_rows = df.sample(n=10)
    no_labels=[10,11,12,13,14]
    random_rows = df[~df['label'].isin(no_labels)].sample(n=10)
    pixel_columns = random_rows.iloc[ :,2:]
    images=[]
    for _, row in pixel_columns.iterrows():
        image = row.values.reshape(3, 32, 32).transpose(1, 2, 0)
        images.append(image)


