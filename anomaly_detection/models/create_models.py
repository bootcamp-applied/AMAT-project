import pickle
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import tensorflow as tf

features_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))


path = '../../data/processed/cifar_10_100.csv'
df = pd.read_csv(path, dtype='int')
grouped = df.groupby('label')

with open('../data/training_set.pickle', 'rb') as file:
    training_set = pickle.load(file)


models = {}

for class_key, class_df in training_set.items():
  images = class_df.values.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
  images = tf.image.resize(images, (96, 96))
  features = features_extractor.predict(preprocess_input(images))
  flatten_features = features.reshape(features.shape[0],-1)
  isolation_forest = IsolationForest(contamination=0.0000001)
  isolation_forest.fit(flatten_features)
  models[class_key] = isolation_forest


models_filename = "../saved_models/isolation_forest_models_low_contamination.joblib"
joblib.dump(models, models_filename)
models_filename = "../saved_models/isolation_forest_models_0_contamination_on_trainset.joblib"
joblib.dump(models, models_filename)





#
# import joblib
# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# import tensorflow as tf
#
# features_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
#
# path = '../../data/processed/cifar_10_100.csv'
# df = pd.read_csv(path, dtype='int')
# grouped = df.groupby('label')
# models = {}
#
# for class_key, class_df in grouped:
#   images = class_df.iloc[:,2:].values.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
#   images = tf.image.resize(images, (96, 96))
#   features = features_extractor.predict(preprocess_input(images))
#   flatten_features = features.reshape(features.shape[0],-1)
#   isolation_forest = IsolationForest(contamination=0)
#   isolation_forest.fit(flatten_features)
#   models[class_key] = isolation_forest
#
# models_filename = "../saved_models/isolation_forest_models_low_contamination.joblib"
# joblib.dump(models, models_filename)
