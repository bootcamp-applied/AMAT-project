# note that the positive class represents the class of the anomalies

import joblib
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

CLASS = 0

features_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

with open('../data/validation_set.pickle', 'rb') as file:
    validation_set = pickle.load(file)

class_df = validation_set[CLASS]
x_positive = class_df[class_df['belongs_to_class'] == 0].drop('belongs_to_class', axis=1)
x_negative = class_df[class_df['belongs_to_class'] == 1].drop('belongs_to_class', axis=1)

x_positive = np.array(x_positive)
x_positive = x_positive.reshape(len(x_positive), 3, 32, 32)
x_positive = x_positive.transpose(0, 2, 3, 1)
x_positive = tf.image.resize(x_positive, (96, 96))
x_positive_features = features_extractor.predict(preprocess_input(x_positive))
x_positive_flatten_features = x_positive_features.reshape(x_positive_features.shape[0], -1)

x_negative = np.array(x_negative)
x_negative = x_negative.reshape(len(x_negative), 3, 32, 32)
x_negative = x_negative.transpose(0, 2, 3, 1)
x_negative = tf.image.resize(x_negative, (96, 96))
x_negative_features = features_extractor.predict(preprocess_input(x_negative))
x_negative_flatten_features = x_negative_features.reshape(x_negative_features.shape[0], -1)

models = joblib.load("../saved_models/isolation_forest_models_low_contamination.joblib")

anomaly_scores_positive = models[CLASS].decision_function(x_positive_flatten_features)
anomaly_scores_negative = models[CLASS].decision_function(x_negative_flatten_features)

plt.hist(anomaly_scores_positive, bins=50, label='Positive Class', alpha=0.5)
plt.hist(anomaly_scores_negative, bins=50, label='Negative Class', alpha=0.5)
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of Anomaly Scores')
plt.show()


