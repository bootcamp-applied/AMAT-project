# 1 (positive class): Represents the anomalous/outlier data points.
# 0 (negative class): Represents the normal/inlines data points.

import joblib
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

CLASS = 12

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

print('dsfas')
print('dsfas')


# Step 5: Calculate and plot ROC curve
X_flatten_features = np.concatenate((x_positive_flatten_features, x_negative_flatten_features), axis=0)
y_positive = np.ones(len(x_positive), dtype=int)
y_negative = np.zeros(len(x_negative), dtype=int)
y_true = np.concatenate((y_positive, y_negative))

# fpr, tpr, thresholds = roc_curve(y_true, -1 * models[CLASS].decision_function(X_flatten_features))
# roc_auc = auc(fpr, tpr)
#
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()



fpr, tpr, thresholds = roc_curve(y_true, -1 * models[CLASS].decision_function(X_flatten_features))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Find threshold for a specific point on the ROC curve
desired_fpr = 0.2  # Choose the desired False Positive Rate
index = np.argmax(fpr >= desired_fpr)  # Find the index closest to the desired FPR
threshold_at_desired_fpr = thresholds[index]

print("Threshold at desired FPR ({}): {:.4f}".format(desired_fpr, threshold_at_desired_fpr))
