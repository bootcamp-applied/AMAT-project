import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


from classification_project.models.CNN1 import CNN

data = pd.read_csv('../../data/processed/cifar_10_100.csv')
print(data.shape)
data = data.iloc[:, 2:]

# Convert the DataFrame to a NumPy array (shape: (60000, 3072))
images = data.to_numpy()

# Reshape each row to 32x32x3 image format
images = images.reshape(-1, 32, 32, 3)


model = CNN.load_cnn_model('../../classification_project/save_models/cnn_model_1.h5').model
preprocessed_images = preprocess_input(images)
feature_layer_index = -2  # Index of the layer before the final dense layer
feature_layer_output = model.layers[feature_layer_index].output

# Create a Keras function to extract features from the given input images
from keras import backend as K

get_features = K.function([model.input], [feature_layer_output])
features = get_features([preprocessed_images])[0]

num_samples = len(features)
num_features = np.prod(features.shape[1:])  # Multiply the dimensions after the first dimension
features_flat = features.reshape(num_samples, num_features)

# Convert the 2D array to a DataFrame
features_df = pd.DataFrame(features_flat)
features_df.columns = features_df.columns.astype(str)

path = '../../data/processed/features_after_CNN.feather'

if os.path.exists(path):
    # os.remove(path)
    existing_df = pd.read_feather(path)
    print(existing_df.shape)
    # Concatenate the existing DataFrame with the new DataFrame
    features_df = pd.concat([existing_df, features_df], ignore_index=True)

# Save the DataFrame to a Feather file
features_df.to_feather(path)




