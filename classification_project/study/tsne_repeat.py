import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from keras.models import Model
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.cnn import CNN

df = pd.read_csv('../../data/processed/cifar-10-100.csv', dtype='int')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras')
loaded_history_model = CNN.load_cnn_history('../saved_model/saved_cnn_history.pkl')

feat_extractor = Model(inputs=loaded_model.model.input,
                       outputs=loaded_model.model.get_layer('dense').output)
features = feat_extractor.predict(x_test)


labels = np.argmax(y_test, axis=1)

tsne = TSNE().fit_transform(features)
tx, ty = tsne[:, 0], tsne[:, 1]
tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

width = 4000
height = 3000
max_dim = 100

# t-SNE with images
full_image = Image.new('RGB', (width, height))
for idx, x in enumerate(x_test):
    tile = Image.fromarray(np.uint8(x * 255))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       resample=Image.Resampling.LANCZOS)
    full_image.paste(tile, (int((width - max_dim) * tx[idx]),
                            int((height - max_dim) * ty[idx])))

plt.imshow(full_image)

# ... (Your existing code)

# Load the new image and its label
new_image_path = 'desk.jpg'  # Replace with the path to your new image
new_label = 15  # Replace with the new label for the image (make sure it is an integer)

# Preprocess the new image
new_image = Image.open(new_image_path).convert('RGB')
new_image = new_image.resize((32, 32))  # Assuming your model takes 32x32 images as input
new_image_array = np.array(new_image) / 255.0  # Normalize the image

# Set the new image to black
new_image_array.fill(0)

# Visualize the new image separately without running t-SNE again
plt.figure(figsize=(5, 5))
plt.imshow(new_image_array)
plt.title('New Image (in Black)')
plt.show()

# Add the new image and label to the test set
x_test_with_new = np.concatenate([x_test, np.expand_dims(new_image_array, axis=0)], axis=0)

# Define the class_names list
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
               'fish', 'people', 'flowers', 'trees', 'fruit and vegetables', 'new_class']

# Convert the new_label to one-hot encoding
new_label_onehot = np.zeros((1, len(class_names)))
new_label_onehot[0, new_label] = 1

# Repeat the one-hot encoded new_label to match the number of samples in x_test
new_label_onehot_repeated = np.repeat(new_label_onehot, x_test.shape[0], axis=0)

# Concatenate the one-hot encoded new_label to y_test
y_test_with_new = np.concatenate([y_test, new_label_onehot_repeated], axis=1)

new_image_features = feat_extractor.predict(np.expand_dims(new_image_array, axis=0))

# 3. Apply t-SNE to visualize the new image in the same plot
features_with_new = np.concatenate([features, new_image_features], axis=0)

# Assuming you have already defined the labels array before adding the new_label
labels_with_new = np.concatenate([labels, np.array([new_label])], axis=0)

tsne = TSNE(n_components=2, random_state=0)
test_representations_2d = tsne.fit_transform(features_with_new)

# Create a plot
# plt.figure(figsize=(10, 10))
# scatter = plt.scatter(test_representations_2d[:, 0], test_representations_2d[:, 1], c=labels_with_new.flatten(), cmap='tab10')
#
# # Create a legend with class names
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
#                'fish', 'people', 'flowers', 'trees', 'fruit and vegetables', 'new_class']
#
# # Update the number of classes in the legend_elements to match the actual number of classes
# legend1 = plt.legend(*scatter.legend_elements(num=len(class_names)), title="Classes")
# plt.gca().add_artist(legend1)
#
# # Convert labels to class names
# class_indices = [int(label.get_text().split("{")[-1].split("}")[0]) for label in legend1.texts]
# for t, class_index in zip(legend1.texts, class_indices):
#     t.set_text(class_names[class_index])
#
# plt.show()
# Create a plot
colors_with_new = np.array([plt.cm.tab10(i) for i in range(len(class_names) + 1)])  # Increase the size by 1 for the new class
colors_with_new[new_label] = [0, 0, 0]

# Create a plot
plt.figure(figsize=(10, 10))

# Plot the existing points
scatter = plt.scatter(test_representations_2d[:-1, 0], test_representations_2d[:-1, 1], c=labels.flatten(), cmap='tab10')

# Plot the new point in black
plt.scatter(test_representations_2d[-1, 0], test_representations_2d[-1, 1], c='k', s=100, marker='X')

# Create a legend with class names
legend1 = plt.legend(*scatter.legend_elements(num=len(class_names) + 1), title="Classes")  # Increase the size by 1 for the new class
plt.gca().add_artist(legend1)

# Convert labels to class names
class_indices = [int(label.get_text().split("{")[-1].split("}")[0]) for label in legend1.texts]
for t, class_index in zip(legend1.texts, class_indices):
    t.set_text(class_names[class_index])

plt.show()