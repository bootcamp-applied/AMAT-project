# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
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
#     return new_image_array
#
# # Load the pre-trained t-SNE model
# tsne_model_path = 'tsne_model.pkl'
# with open(tsne_model_path, 'rb') as f:
#     tsne = pickle.load(f)
#
# # Load the pre-trained CNN model
# loaded_model = CNN.load_cnn_model('../saved_model/cnn_model_all_data.keras')
# loaded_history_model = CNN.load_cnn_history('../saved_model/cnn_history_all_data.pkl')
#
# # Create a feature extractor model
# feat_extractor = Model(inputs=loaded_model.model.input, outputs=loaded_model.model.get_layer('dense').output)
#
# # Preprocess the new image
# new_image_path = 'bird.jpg'  # Replace with the path to your new image
# new_image_size = (32, 32)  # Assuming your model takes 32x32 images as input
# new_image_array = preprocess_new_image(new_image_path, new_image_size)
#
# # Obtain the feature representation of the new image using the pre-trained CNN model
# new_image_features = feat_extractor.predict(np.expand_dims(new_image_array, axis=0))
#
# # Load the dataset and prepare the data
# df = pd.read_csv('../../data/processed/cifar-10-100-augmentation.csv', dtype='int')
# preprocessing = Preprocessing(df)
# preprocessing.prepare_data()
# x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)
#
# # Extract features from the test set
# features = feat_extractor.predict(x_test)
#
# labels = np.argmax(y_test, axis=1)
#
# # Get the unique classes from the labels array
# unique_classes = np.unique(labels)
#
# # Create a list of class names based on the unique classes in the labels array
# #class_names = [f'Class {cls}' for cls in unique_classes]
# features_with_new = np.concatenate([features, new_image_features], axis=0)
# # Apply the pre-trained t-SNE model to the combined features
# test_representations_2d = tsne.fit_transform(features_with_new)
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fish', 'people', 'flowers', 'trees', 'fruit and vegetables']
#
# # Create a plot
# plt.figure(figsize=(10, 10))
#
# # Plot the existing points with class names
# scatter = plt.scatter(test_representations_2d[:-1, 0], test_representations_2d[:-1, 1], c=labels, cmap='tab20')
#
# for i, class_name in enumerate(class_names):
#     # Get the indices of points belonging to the current class
#     indices = np.where(labels == i)[0]
#     # Calculate the mean coordinates for the current class
#     mean_x = np.mean(test_representations_2d[indices, 0])
#     mean_y = np.mean(test_representations_2d[indices, 1])
#
#     # Plot the class name at the mean coordinates
#     plt.text(mean_x, mean_y, class_name,
#              fontsize=12, weight='bold', alpha=0.75, color=scatter.to_rgba(i))
#
# # Plot the new point in black
# plt.scatter(test_representations_2d[-1, 0], test_representations_2d[-1, 1], c='black', s=100, marker='X',
#             label='New Image')
#
# # Create a legend with class names
# legend1 = plt.legend(*scatter.legend_elements(num=15), title="Classes")
# plt.gca().add_artist(legend1)
#
# # Show the plot
# plt.show()


# import numpy as np
# from PIL import Image
# from keras.models import Model
# import pickle
# import matplotlib.pyplot as plt
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
#     return new_image_array
#
# # Load the pre-trained t-SNE model
# tsne_model_path = 'tsne_model.pkl'
# with open(tsne_model_path, 'rb') as f:
#     tsne = pickle.load(f)
#
# # Load the pre-trained CNN model
# loaded_model = CNN.load_cnn_model('../saved_model/cnn_model_all_data.keras')
# loaded_history_model = CNN.load_cnn_history('../saved_model/cnn_history_all_data.pkl')
#
# # Create a feature extractor model
# feat_extractor = Model(inputs=loaded_model.model.input, outputs=loaded_model.model.get_layer('dense').output)
#
# # Preprocess the new image
# new_image_path = 'bird.jpg'  # Replace with the path to your new image
# new_image_size = (32, 32)  # Assuming your model takes 32x32 images as input
# new_image_array = preprocess_new_image(new_image_path, new_image_size)
#
# # Obtain the feature representation of the new image using the pre-trained CNN model
# new_image_features = feat_extractor.predict(np.expand_dims(new_image_array, axis=0))
#
# # Load the pre-computed features for the test set
# features_path = 'test_features.pkl'
# with open(features_path, 'rb') as f:
#     features = pickle.load(f)
#
# # Combine the extracted features of the test set with the feature representation of the new image
# features_with_new = np.concatenate([features, new_image_features], axis=0)
#
# # Apply the pre-trained t-SNE model to the combined features
# test_representations_2d = tsne.fit_transform(features_with_new)
#
# # Rest of the code to visualize the points...
#
# labels = np.argmax(y_test, axis=1)
#
# # Get the unique classes from the labels array
# unique_classes = np.unique(labels)
#
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fish', 'people', 'flowers', 'trees', 'fruit and vegetables']
#
# # Create a plot
# plt.figure(figsize=(10, 10))
#
# # Plot the existing points with class names
# scatter = plt.scatter(test_representations_2d[:-1, 0], test_representations_2d[:-1, 1], c=labels, cmap='tab20')
#
# for i, class_name in enumerate(class_names):
#     # Get the indices of points belonging to the current class
#     indices = np.where(labels == i)[0]
#     # Calculate the mean coordinates for the current class
#     mean_x = np.mean(test_representations_2d[indices, 0])
#     mean_y = np.mean(test_representations_2d[indices, 1])
#
#     # Plot the class name at the mean coordinates
#     plt.text(mean_x, mean_y, class_name,
#              fontsize=12, weight='bold', alpha=0.75, color=scatter.to_rgba(i))
#
# # Plot the new point in black
# plt.scatter(test_representations_2d[-1, 0], test_representations_2d[-1, 1], c='black', s=100, marker='X',
#             label='New Image')
#
# # Create a legend with class names
# legend1 = plt.legend(*scatter.legend_elements(num=15), title="Classes")
# plt.gca().add_artist(legend1)
#
# # Show the plot
# plt.show()

import numpy as np
from PIL import Image
from keras.models import Model
import pickle
import matplotlib.pyplot as plt

from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN

def preprocess_new_image(image_path, image_size):
    # Load the image from the given path and convert it to RGB mode
    new_image = Image.open(image_path).convert('RGB')

    # Resize the image to the specified size
    new_image = new_image.resize(image_size)

    # Convert the image to a NumPy array and normalize the pixel values to the range [0, 1]
    new_image_array = np.array(new_image) / 255.0

    return new_image_array

# Load the pre-trained t-SNE model
tsne_model_path = 'tsne_model.pkl'
with open(tsne_model_path, 'rb') as f:
    tsne = pickle.load(f)

# Load the pre-trained CNN model
loaded_model = CNN.load_cnn_model('../saved_model/cnn_model_all_data.keras')
loaded_history_model = CNN.load_cnn_history('../saved_model/cnn_history_all_data.pkl')

# Create a feature extractor model
feat_extractor = Model(inputs=loaded_model.model.input, outputs=loaded_model.model.get_layer('dense').output)

# Preprocess the new image
new_image_path = 'efrat.jpg'  # Replace with the path to your new image
new_image_size = (32, 32)  # Assuming your model takes 32x32 images as input
new_image_array = preprocess_new_image(new_image_path, new_image_size)

# Obtain the feature representation of the new image using the pre-trained CNN model
new_image_features = feat_extractor.predict(np.expand_dims(new_image_array, axis=0))

# Load the pre-computed features and labels for the test set
test_data_path = 'test_data.pkl'
with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

features, y_test = test_data  # Extract features and labels from the loaded tuple

# Combine the extracted features of the test set with the feature representation of the new image
features_with_new = np.concatenate([features, new_image_features], axis=0)

# Apply the pre-trained t-SNE model to the combined features
test_representations_2d = tsne.fit_transform(features_with_new)

# Rest of the code to visualize the points...

labels = np.argmax(y_test, axis=1)

# Get the unique classes from the labels array
unique_classes = np.unique(labels)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fish', 'people', 'flowers', 'trees', 'fruit and vegetables']

# Create a plot
plt.figure(figsize=(10, 10))

# Plot the existing points with class names
scatter = plt.scatter(test_representations_2d[:-1, 0], test_representations_2d[:-1, 1], c=labels, cmap='tab20')

for i, class_name in enumerate(class_names):
    # Get the indices of points belonging to the current class
    indices = np.where(labels == i)[0]
    # Calculate the mean coordinates for the current class
    mean_x = np.mean(test_representations_2d[indices, 0])
    mean_y = np.mean(test_representations_2d[indices, 1])

    # Plot the class name at the mean coordinates
    plt.text(mean_x, mean_y, class_name,
             fontsize=12, weight='bold', alpha=0.75, color=scatter.to_rgba(i))

# Plot the new point in black
plt.scatter(test_representations_2d[-1, 0], test_representations_2d[-1, 1], c='black', s=100, marker='X',
            label='New Image')

# Create a legend with class names
legend1 = plt.legend(*scatter.legend_elements(num=15), title="Classes")
plt.gca().add_artist(legend1)

# Show the plot
plt.show()
