import base64
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from classification_project.models.CNN1 import CNN1
from classification_project.utils.handling_new_image import NewImage
import base64
from PIL import Image
import io

new_image = None
label = None


def format_image(image):
    global new_image
    _, image = image.split(',')
    image = base64.b64decode(image)
    # Convert decoded image data to numpy array
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    handler = NewImage()
    image, _ = handler.image_handle(image)
    # flat the image
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nor_image = new_image.astype('float32') / 255
    final_image = nor_image.reshape((1, 32, 32, 3))  # .transpose(0, 2, 3, 1)
    return final_image


def predict_label(image):
    global label
    # load the model
    loaded_model = CNN1.load_cnn_model('../saved_model/saved_cnn_model.keras').model
    probabilities = loaded_model.predict(image)
    label = np.argmax(probabilities)
    label_category = convert_to_category()
    return label_category


def convert_to_category():
    map_label = '../utils/dict.json'
    with open(map_label, 'r') as f:
        label_dict = json.load(f)
    label_category = next((val for key, val in label_dict.items() if key == str(label)), None)
    return label_category


def similar_images():
    df = pd.read_csv('../../data/processed/cifar-100.csv')
    df_class = df[df['label'] == label]
    features_class = df_class.iloc[:, 2:]
    # features_array = features_class.to_numpy()
    flattened_image = np.ravel(new_image)
    distances = []
    for row in features_class.values:
        distances.append(np.sum(np.abs(row - flattened_image)))
    sorted_indices = np.argsort(distances)
    four_closest_indices = sorted_indices[:4]
    four_closest_vectors = features_class.iloc[four_closest_indices]
    reshaped_transposed_list = [row.reshape(3, 32, 32).transpose(1, 2, 0) for row in four_closest_vectors.values]
    # four_closest_vectors = four_closest_vectors.values.reshape(3, 32, 32).transpose(1, 2, 0)
    # closest = four_closest_vectors.values[0].reshape(3, 32, 32).transpose(1, 2, 0)
    # plt.imshow(closest)
    # plt.show()
    return reshaped_transposed_list


def encode_image(image_vector):
    # Reshape the image_vector to (32, 32, 3) if it's in the correct shape
    image_array = image_vector.reshape(32, 32, 3)
    # Convert the image_array (a NumPy array) to a PIL image
    img = Image.fromarray(image_array.astype('uint8'))
    # Convert the PIL image to a base64-encoded string
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()
    # plt.imshow(base64_image)
    # plt.show()
    return base64_image
