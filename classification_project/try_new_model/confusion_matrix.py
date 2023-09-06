import pandas as pd
import random
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix
# 1 15 class
# 0 anomaly

df = pd.read_feather('../../data/processed/cifar_10_100_all.feather')

X0 = df[df['label'] == 15]
X0 = X0.sample(n=1000)

filtered_df = df[df['label'] != 15]

X1 = filtered_df.sample(n=1000)
#
# condition = df['label'] == 15
#
# # Use boolean indexing to filter the DataFrame and keep only rows where the condition is not met
# df.drop(df[condition].index, inplace=True)
#
# X0['label'] = 0
# df['label'] = 1
#
# X = pd.concat([X0, df], ignore_index=True)
# print(X.head())
#
# model = load_model('./saved_models/keras_all_data_trained_model_16_classes.h5')
#
# y_pred = model.predict(X)
# y_pred = [1 if y < 15 else 0 for y in y_pred]
# y_true = X['label']
#
# confusion_matrix(y_true, y_pred)
