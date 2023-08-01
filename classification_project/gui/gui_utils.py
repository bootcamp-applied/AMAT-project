import base64
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from classification_project.models.CNN1 import CNN1
from classification_project.utils.handling_new_image import NewImage


def format_image(image):
    _, image = image.split(',')
    image = base64.b64decode(image)
    # Convert decoded image data to numpy array
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    handler = NewImage()
    image, _ = handler.image_handle(image)
    # flat the image
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # flat_img = new_image.transpose(2, 0, 1).reshape(1, -1)
    # normalize the values
    nor_image = new_image.astype('float32') / 255
    nor_image = nor_image.reshape(1, 32, 32, 3)
    plt.imshow(new_image)
    plt.show()
    return nor_image


def predict_label(image):
    # load the model
    loaded_model = CNN1.load_cnn_model('../saved_model/saved_cnn_model.keras').model
    probabilities = loaded_model.predict(image)
    label = np.argmax(probabilities)
    label_category = convert_to_category(label)
    return label_category


def convert_to_category(label):
    map_label = '../utils/dict.json'
    with open(map_label, 'r') as f:
        label_dict = json.load(f)
    label_category = next((val for key, val in label_dict.items() if key == str(label)), None)
    return label_category

