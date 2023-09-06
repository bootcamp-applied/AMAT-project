# import pandas as pd
# import numpy as np
# from sklearn.manifold import TSNE
# from PIL import Image
# from keras.models import Model
# import pickle
#
# from classification_project.preprocessing.preprocessing import Preprocessing
# from classification_project.models.CNN1 import CNN1
#
# def preprocess_new_image(image_path, image_size):
#     # Load the image from the given path and convert it to RGB mode
#     new_image = Image.open(image_path).convert('RGB')
#
#     # Resize the image to the specified size
#     new_image = new_image.resize(image_size)
#
#     # Convert the image to a NumPy array and normalize the pixel values to the range [0, 1]
#     new_image_array = np.array(new_image) / 255.0
#
#     return new_image_array  # Return only the preprocessed image
#
# # Load the dataset and prepare the data
# df = pd.read_csv('../../data/processed/cifar-10-100-augmentation.csv', dtype='int')
# preprocessing = Preprocessing(df)
# preprocessing.prepare_data()
# x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)
#
# # Load the pre-trained CNN model
# loaded_model = CNN.load_cnn_model('../saved_model/cnn_model_all_data.keras')
# loaded_history_model = CNN.load_cnn_history('../saved_model/cnn_history_all_data.pkl')
#
# # Create a feature extractor model
# feat_extractor = Model(inputs=loaded_model.model.input, outputs=loaded_model.model.get_layer('dense').output)
#
# # Extract features from the test set
# features = feat_extractor.predict(x_test)
#
# # Save the features to a file
# features_path = 'test_features.pkl'
# with open(features_path, 'wb') as f:
#     pickle.dump(features, f)
#
# # Train t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# tsne.fit(features)
#
# # Save the t-SNE model to a file
# tsne_model_path = 'tsne_model.pkl'
# with open(tsne_model_path, 'wb') as f:
#     pickle.dump(tsne, f)

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image
from keras.models import Model
import pickle
from keras.models import load_model

from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN1

def preprocess_new_image(image_path, image_size):
    # Load the image from the given path and convert it to RGB mode
    new_image = Image.open(image_path).convert('RGB')

    # Resize the image to the specified size
    new_image = new_image.resize(image_size)

    # Convert the image to a NumPy array and normalize the pixel values to the range [0, 1]
    new_image_array = np.array(new_image) / 255.0

    return new_image_array  # Return only the preprocessed image

# Load the dataset and prepare the data
df = pd.read_csv('../../data/processed/cifar-10-100-augmentation.csv', dtype='int')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

# Load the pre-trained CNN model
loaded_model = CNN1.load_cnn_model('../saved_models/saved_cnn_model.keras')
loaded_history_model = CNN1.load_cnn_history('../saved_models/saved_cnn_model.pkl')

# Create a feature extractor model
feat_extractor = Model(inputs=loaded_model.model.input, outputs=loaded_model.model.get_layer('dense').output)

# Extract features from the test set
features = feat_extractor.predict(x_test)

# Save the features and labels to a file
test_data = (features, y_test)  # Store both features and labels as a tuple
test_data_path = 'test_data2.pkl'

with open(test_data_path, 'wb') as f:
    pickle.dump(test_data, f)

# Train t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne.fit(features)

# Save the t-SNE model to a file
tsne_model_path = 'tsne_model2.pkl'
with open(tsne_model_path, 'wb') as f:
    pickle.dump(tsne, f)


# images = df.iloc[:, 2:]
# images = np.array(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
# images = images.astype('float') / 255
# y_test = df['label']
# cnn_model = load_model('../saved_models/cnn_model_all_data.keras')
# feat_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('dense_1').output)
#
# features = feat_extractor.predict(images)
#
# # Save the features and labels to a file
# test_data = (features, y_test)  # Store both features and labels as a tuple
# test_data_path = 'test_data_activation_5.pkl'
