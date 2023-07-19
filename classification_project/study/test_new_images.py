import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../data/processed/cifar-10-100.csv')

last_img = df.iloc[-1,:][2:].values.reshape(3,32,32).transpose(1,2,0)

plt.imshow(last_img)
plt.show()
