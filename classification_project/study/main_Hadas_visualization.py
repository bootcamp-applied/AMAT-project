import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.visualization.visualization import Visualization

df = pd.read_csv('../../data/processed/cifar-10.csv')
preprocessing = Preprocessing(df)

train_images, y_train, test_images, y_test = preprocessing.split_data(include_validation=False)

# Resize the images to the size expected by MobileNetV2
train_images = tf.image.resize(train_images, (96, 96))
test_images = tf.image.resize(test_images, (96, 96))


# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Extract features from the images
train_features = base_model.predict(train_images)  # Preprocessing required by MobileNetV2
test_features = base_model.predict(test_images)

# Flatten the features into 1D vectors
flat_train_features = train_features.reshape((-1, np.prod(train_features.shape[1:])))
flat_test_features = test_features.reshape((-1, np.prod(test_features.shape[1:])))


visualization = Visualization()


visualization.TSNE(X, y)
# visu.Pareto(y)




