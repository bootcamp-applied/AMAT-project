import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras.models import Model
import pickle
from openTSNE import affinity
from openTSNE import initialization
import openTSNE
import matplotlib.pyplot as plt
from classification_project.models.CNN1 import CNN1


# def preprocess_new_image(image_url, image_size):
#     # Download the image from the URL with SSL certificate verification disabled
#     response = requests.get(image_url, verify=False)
#     new_image = Image.open(BytesIO(response.content)).convert('RGB')
#
#     # Resize the image to the specified size
#     new_image = new_image.resize(image_size)
#
#     # Convert the image to a NumPy array and normalize the pixel values to the range [0, 1]
#     new_image_array = np.array(new_image) / 255.0
#
#     return new_image_array

# Load the pre-trained t-SNE model

def preprocess_new_image(image_path, image_size):
    # Open the image from the local path
    new_image = Image.open(image_path).convert('RGB')

    # Resize the image to the specified size
    new_image = new_image.resize(image_size)

    # Convert the image to a NumPy array and normalize the pixel values to the range [0, 1]
    new_image_array = np.array(new_image) / 255.0

    return new_image_array

tsne_model_path = 'tsne_model2.pkl'

with open(tsne_model_path, 'rb') as f:
    # tsne = pickle.load(f)
    tsne = pickle.load(f)

from keras.models import load_model
cnn_model = load_model('../saved_models/saved_cnn_model.keras')
feat_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('dense').output)


# Preprocess the new image
new_image_url = "../utils/horse14 (2).jpg"
new_image_size = (32, 32)
new_image_array = preprocess_new_image(new_image_url, new_image_size)

# Obtain the feature representation of the new image using the pre-trained CNN model
new_image_features = feat_extractor.predict(np.expand_dims(new_image_array, axis=0))

# Load the pre-computed features and labels for the test set
test_data_path = 'test_data2.pkl'

with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

features, y_test = test_data  # Extract features and labels from the loaded tuple

# Combine the extracted features of the test set with the feature representation of the new image
features_with_new = np.concatenate([features, new_image_features], axis=0)

# Apply the pre-trained t-SNE model to the combined features
test_representations_2d = tsne.fit_transform(features_with_new)
#embedding_test = tsne.prepare_partial(x_test)
#test_representations_2d = tsne.fit_transform(new_image_features,initialization=initial_embedding)

# labels = np.argmax(y_test, axis=1)
labels = np.array(y_test)

# Get the unique classes from the labels array
unique_classes = np.unique(labels)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fish', 'people', 'flowers', 'trees', 'fruit and vegetables']

# Create a plot
plt.figure(figsize=(10, 10))

# Plot the existing points with class names
scatter = plt.scatter(test_representations_2d[:-1, 0], test_representations_2d[:-1, 1], c=labels, cmap='tab20')

# Extract the handles and labels from the scatter plot
handles, labels = scatter.legend_elements(num=15)

# Plot the class names at the corresponding positions
for i, class_name in enumerate(class_names):
    class_color = handles[i].get_color()
    class_mean_x = test_representations_2d[labels == i, 0].mean()
    class_mean_y = test_representations_2d[labels == i, 1].mean()
    if not np.isnan(class_mean_x) and not np.isnan(class_mean_y):
        plt.text(class_mean_x, class_mean_y, class_name,
                 fontsize=12, weight='bold', alpha=0.75, color=class_color)

# Plot the new point in black
plt.scatter(test_representations_2d[-1, 0], test_representations_2d[-1, 1], c='black', s=100, marker='X',
            label='New Image')

# Create a legend with class names
legend1 = plt.legend(handles, class_names, title="Classes")
plt.gca().add_artist(legend1)

# Show the plot
plt.show()

