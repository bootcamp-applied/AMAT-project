# import pandas as pd
# import numpy as np
# from sklearn.manifold import TSNE
# from PIL import Image
# from keras.models import Model
# import pickle
#
# from classification_project.preprocessing.preprocessing import Preprocessing
# from classification_project.models.CNN1 import CNN
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
# # Save the features and labels to a file
# test_data = (features, y_test)  # Store both features and labels as a tuple
# test_data_path = 'test_data.pkl'
# with open(test_data_path, 'wb') as f:
#     pickle.dump(test_data, f)
#
# # Train t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# initial_embedding =tsne.fit(features)
#
# # Save the t-SNE model to a file
# tsne_model_path = 'tsne_model.pkl'
# with open(tsne_model_path, 'wb') as f:
#     # pickle.dump(tsne, f)
#     pickle.dump((tsne, initial_embedding), f)




import pandas as pd
import numpy as np
from openTSNE import TSNE
from openTSNE import initialization
from openTSNE import TSNEEmbedding, affinity
from PIL import Image
from keras.models import Model
import pickle

from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN

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
loaded_model = CNN.load_cnn_model('../saved_model/cnn_model_all_data.keras')
loaded_history_model = CNN.load_cnn_history('../saved_model/cnn_history_all_data.pkl')

# feat_extractor = Model(inputs=loaded_model.model.input, outputs=loaded_model.model.get_layer('dense').output)
#
# # Extract features from the test set
# features = feat_extractor.predict(x_test)
#
# # Save the features and labels to a file
# test_data = (features, y_test)  # Store both features and labels as a tuple
# test_data_path = 'test_data1.pkl'
# with open(test_data_path, 'wb') as f:
#     pickle.dump(test_data, f)
#
# # Create an affinity matrix for the training data
# # affinities_train = affinity.PerplexityBasedNN(
# #     features,
# #     perplexity=30,
# #     metric="euclidean",
# #     n_jobs=8,
# #     random_state=0,
# #     verbose=True,
# # )
# #
# # # Create a t-SNE embedding using the affinity matrix
# # embedding_train = TSNEEmbedding(
# #     affinities=affinities_train,  # Provide the affinities parameter
# #     negative_gradient_method="fft",
# #     n_jobs=8,
# #     verbose=True,
# # )
#
#
# # ... (Rest of your imports and code)
#
# # Create an affinity matrix for the training data
# affinities_train = affinity.PerplexityBasedNN(
#     features,
#     perplexity=30,
#     metric="euclidean",
#     n_jobs=8,
#     random_state=0,
#     verbose=True,
# )
#
# # Create a t-SNE embedding using the affinity matrix
# embedding_train = TSNEEmbedding(
#     affinities=affinities_train,
#     negative_gradient_method="fft",
#     n_jobs=8,
#     verbose=True,
# )
#
# # Optimize the embedding (this performs the actual t-SNE computation)
# embedding_train.optimize(n_iter=250)  # Adjust the number of iterations as needed
#
# # Save the t-SNE model to a file
# tsne_model_path = 'tsne_model1.pkl'
# with open(tsne_model_path, 'wb') as f:
#     pickle.dump(embedding_train, f)
#
feat_extractor = Model(inputs=loaded_model.model.input, outputs=loaded_model.model.get_layer('dense').output)

# Extract features from the test set
features = feat_extractor.predict(x_test)

# Save the features and labels to a file
test_data = (features, y_test)  # Store both features and labels as a tuple
test_data_path = 'test_data1.pkl'
with open(test_data_path, 'wb') as f:
    pickle.dump(test_data, f)

# Create an affinity matrix for the training data
affinities_train = affinity.PerplexityBasedNN(
    features,
    perplexity=30,  # Adjust the perplexity as needed
    metric="euclidean",
    n_jobs=8,
    random_state=0,  # Set your desired random state
    verbose=True,
)
init_train = initialization.pca(features, random_state=42)

# Create a t-SNE embedding using the affinity matrix
embedding_train = TSNEEmbedding(
    init_train,
    affinities=affinities_train,
    negative_gradient_method="fft",
    n_jobs=8,
    verbose=True,
)

# embedding_train = TSNEEmbedding(
#     init_train,
#     affinities_train,
#     negative_gradient_method="fft",
#     n_jobs=8,
#     verbose=True,
# )

# Optimize the embedding (this performs the actual t-SNE computation)
embedding_train.optimize(n_iter=250)  # Adjust the number of iterations as needed

# Save the t-SNE model to a file
tsne_model_path = 'tsne_model1.pkl'
with open(tsne_model_path, 'wb') as f:
    pickle.dump(embedding_train, f)
