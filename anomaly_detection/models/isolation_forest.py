
import pandas as pd

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import tensorflow as tf

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
path = '../../data/processed/cifar-10-100.csv'
df = pd.read_csv(path, dtype='int')
images = df.iloc[:,2:].values.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
images = tf.image.resize(images, (96, 96))
feature = base_model.predict(preprocess_input(images))
print('hi')



from sklearn.ensemble import IsolationForest

grouped = df.groupby('label')
models = {}

for name, data in grouped:
  images = data.iloc[:,2:].values.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
  images = tf.image.resize(images, (96, 96))
  feature = base_model.predict(preprocess_input(images))
  isolation_forest = IsolationForest(contamination=0.05)
  isolation_forest.fit(feature.reshape(feature.shape[0],-1))
  models[name] = isolation_forest
