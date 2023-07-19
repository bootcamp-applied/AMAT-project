import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.visualization.visualization import Visualization

df = pd.read_csv('../../data/processed/cifar-10.csv')
preprocessing = Preprocessing(df)
x_test, y_train, x_test, y_test = preprocessing.split_data(include_validation=False)
train_images = x_test
test_images = x_test
train_labels = y_train
test_labels = y_test

# Resize the images to the size expected by MobileNetV2
test_images = tf.image.resize(test_images, (96, 96))

# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Extract features from the images
test_features = base_model.predict(preprocess_input(test_images))


# Flatten the features into 1D vectors
flat_test_features = test_features.reshape((-1, np.prod(test_features.shape[1:])))

# Convert the labels to proper dim
# test_labels = np.reshape(-1,)
test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

visualization = Visualization()
visualization.TSNE(flat_test_features, test_labels)
# visu.Pareto(y)




