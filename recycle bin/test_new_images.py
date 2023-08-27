import pandas as pd
import cv2

df = pd.read_csv('../data/processed/cifar_10_100.csv', nrows=3)

first_img = df.iloc[1,:][2:].values.reshape(3,32,32).transpose(1,2,0)

cv2.imwrite('new_image.jpg',first_img)
