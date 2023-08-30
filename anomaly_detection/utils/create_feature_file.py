from classification_project.models.CNN1 import CNN1
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

# Load the CNN model
cnn_model = CNN1.load_cnn_model('../../classification_project/saved_models/cnn_model_all_data.keras')

# Create a feature extractor model using a specific layer
feat_extractor = Model(inputs=cnn_model.model.input, outputs=cnn_model.model.get_layer('dense_1').output)

# Read the data from the feather file
df = pd.read_feather('../data/df.feather')

images = np.array(df.iloc[:, 2:]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Normalize the images
normalized_images = images.astype('float32') / 255

# Extract features using the feature extractor model
features = feat_extractor.predict(normalized_images)

# Reshape the features for saving to a DataFrame
flatten_features = features.reshape(features.shape[0], -1)

# Create a DataFrame with the flattened features and labels
feature_df = pd.DataFrame(flatten_features)
feature_df.insert(0, 'label', df['label'])
feature_df.columns = feature_df.columns.astype(str)

# Save the feature DataFrame to a feather file
feature_df.to_feather('../data/features_6.feather')
