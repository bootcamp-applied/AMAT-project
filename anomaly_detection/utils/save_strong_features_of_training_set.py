import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.image import resize

data = pd.read_feather('../../data/processed/cifar_10_100_augmentation.feather')
data = data.iloc[:, 2:]

# Convert the DataFrame to a NumPy array (shape: (60000, 3072))
images = data.to_numpy()

# Reshape each row to 32x32x3 image format
images = images.reshape(-1, 32, 32, 3)

# Resize the images to the required input shape (96x96x3)
images_resized = []
for img in images:
    img_resized = resize(img, [96, 96])
    images_resized.append(img_resized)

images_resized = np.array(images_resized)

# Preprocess the images
images_preprocessed = preprocess_input(images_resized)

# Create the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Predict using the preprocessed images
features = model.predict(images_preprocessed)
num_samples = len(features)
num_features = np.prod(features.shape[1:])  # Multiply the dimensions after the first dimension
features_flat = features.reshape(num_samples, num_features)

# Convert the 2D array to a DataFrame
features_df = pd.DataFrame(features_flat)
features_df.columns = features_df.columns.astype(str)

path = '../../data/processed/features.feather'

if os.path.exists(path):
    # If the file exists, delete it
    os.remove(path)

# Save the DataFrame to a Feather file
features_df.to_feather(path)
