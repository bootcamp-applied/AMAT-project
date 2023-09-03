import base64
import json
import matplotlib.pyplot as plt
import cv2
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from classification_project.models.CNN1 import CNN1 as CNN
from classification_project.utils.handling_new_image import NewImage
import base64
from PIL import Image
import io
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.image import resize
from keras import backend as K
from tensorflow.keras.models import load_model, Model

new_image = None
# label = None
probabilities = None
similar_images_label = None

def format_image(image):
    global new_image
    _, image = image.split(',')
    image = base64.b64decode(image)
    # Convert decoded image data to numpy array
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    handler = NewImage()
    image = handler.image_handle(image)
    # flat the image
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nor_image = new_image.astype('float32') / 255
    final_image = nor_image.reshape((1, 32, 32, 3))  # .transpose(0, 2, 3, 1)
    return final_image


def predict_label(image):
    # global label
    global probabilities
    # load the model
    loaded_model = CNN.load_cnn_model('../saved_models/cnn_model_all_data.keras').model
    probabilities = loaded_model.predict(image)
    label = np.argmax(probabilities)
    label_category = convert_to_category(label)
    return label_category

def is_anomalysis(image):
    print("1")
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
        "fish", "people", "flowers", "trees", "fruit and vegetables"
    ]
    models = joblib.load('../../anomaly_detection/saved_models/best_models.pkl')
    thresholds = joblib.load('../../anomaly_detection/saved_models/best_thresholds.pkl')
    model = load_model("../saved_models/cnn_model_1.h5")
    feature_extractor = Model(input=model.input, output=model.get_layer('dense_1').output)
    features_of_image=feature_extractor.predict(image)
    belongs = []
    results = {}
    for class_idx in range(15):
      score = models[class_idx].decision_function(features_of_image)
      if score[0] > thresholds[class_idx]:
        belongs.append(class_names[class_idx])
    print(belongs)
    if len(belongs) > 0:
        predict_label(image)
    else:
        return "anomaly"

<<<<<<< HEAD


=======
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
def convert_to_category(label):
    map_label = '../utils/dict.json'
    with open(map_label, 'r') as f:
        label_dict = json.load(f)
    label_category = next((val for key, val in label_dict.items() if key == str(label)), None)
    return label_category


# def similar_images_from_base_csv():
#     df = pd.read_csv('../../data/processed/cifar-100.csv')
#     df_class = df[df['label'] == label]
#     features_class = df_class.iloc[:, 2:]
#     # features_array = features_class.to_numpy()
#     flattened_image = np.ravel(new_image)
#     distances = []
#     for row in features_class.values:
#         distances.append(np.sum(np.abs(row - flattened_image)))
#     sorted_indices = np.argsort(distances)
#     four_closest_indices = sorted_indices[:4]
#     four_closest_vectors = features_class.iloc[four_closest_indices]
#     reshaped_transposed_list = [row.reshape(3, 32, 32).transpose(1, 2, 0) for row in four_closest_vectors.values]
#     # four_closest_vectors = four_closest_vectors.values.reshape(3, 32, 32).transpose(1, 2, 0)
#     # closest = four_closest_vectors.values[0].reshape(3, 32, 32).transpose(1, 2, 0)
#     # plt.imshow(closest)
#     # plt.show()
#     return reshaped_transposed_list


def similar_images_mobileV2_features():
    global new_image
    # Load the DataFrame with flattened images
    data_features = pd.read_feather('../../data/processed/features_base_data.feather')
    data = pd.read_feather('../../data/processed/cifar_10_100.feather')
    data = data.iloc[:, 2:]
    # Convert the DataFrame to a NumPy array

    new_image = resize(new_image, [96, 96])

    new_image = np.array(new_image)
    # Preprocess the images
    reshaped_image_array = new_image.reshape(1, 96, 96, 3)
    new_image_preprocessed = preprocess_input(reshaped_image_array)
    # Create the MobileNetV2 model
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    new_image_features = model.predict(new_image_preprocessed)
    new_image_features = new_image_features.flatten()
    distances = []
    for row in data_features.values:
        distances.append(np.sum(np.abs(row - new_image_features)))

    sorted_indices = np.argsort(distances)
    four_closest_indices = sorted_indices[:4]
    # te read data nit features
    four_closest_vectors = data.iloc[four_closest_indices]
    return four_closest_vectors
    # for i, image in enumerate(four_closest_vectors.values):
    #     plt.subplot(1, 4, i+1)
    #     image = image.reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape the flattened image to the original shape (32x32x3)
    #     plt.imshow(image)
    #     plt.axis('off')
    #
    # plt.show()

    # reshaped_transposed_list = [row.reshape(3, 32, 32).transpose(1, 2, 0) for row in four_closest_vectors]
    # # four_closest_vectors = four_closest_vectors.values.reshape(3, 32, 32).transpose(1, 2, 0)
    # print(features.shape)
    # return reshaped_transposed_list


def similar_images_cnn_features():
    global new_image
    # Load the DataFrame with flattened images
    data_features = pd.read_feather('../../data/processed/features_after_CNN.feather')
    data = pd.read_feather('../../data/processed/cifar_10_100.feather')
    data = data.iloc[:, 2:]
    # Preprocess the images
    reshaped_image_array = new_image.reshape(1, 32, 32, 3)
    model = CNN.load_cnn_model('../save_models/cnn_model_1.h5').model
    preprocessed_image = preprocess_input(reshaped_image_array)
    feature_layer_index = -2  # Index of the layer before the final dense layer
    feature_layer_output = model.layers[feature_layer_index].output
    # Create a Keras function to extract features from the given input images
    get_features = K.function([model.input], [feature_layer_output])
    new_image_features = get_features([preprocessed_image])[0]
    distances = []
    for row in data_features.values:
        distances.append(np.sum(np.abs(row - new_image_features)))
    sorted_indices = np.argsort(distances)
    # similarity_scores = cosine_similarity(new_image_features, data_features)
    # closest_indices = np.argsort(similarity_scores[0])[::-1][:4]
    four_closest_indices = sorted_indices[:4]
    # te read data features
    four_closest_vectors = data.iloc[four_closest_indices]
    return four_closest_vectors


def similar_images_cnn_features_cosine():
    global new_image
    # Load the DataFrame with flattened images
    data_features = pd.read_feather('../../data/processed/features_after_CNN.feather')
    data = pd.read_feather('../../data/processed/cifar_10_100.feather')
    data = data.iloc[:, 2:]
    # Preprocess the images
    reshaped_image_array = new_image.reshape(1, 32, 32, 3)
    model = CNN.load_cnn_model('classification_project/saved_models/cnn_model_1.h5').model
    preprocessed_image = preprocess_input(reshaped_image_array)
    feature_layer_index = -2  # Index of the layer before the final dense layer
    feature_layer_output = model.layers[feature_layer_index].output
    # Create a Keras function to extract features from the given input images
    get_features = K.function([model.input], [feature_layer_output])
    new_image_features = get_features([preprocessed_image])[0]
    similarity_scores = cosine_similarity(new_image_features, data_features)
    four_closest_indices = np.argsort(similarity_scores[0])[::-1][:4]
    # te read data features
    four_closest_vectors = data.iloc[four_closest_indices]
    return four_closest_vectors


def similar_images_using_potential():
    global new_image
    global similar_images_label
    data_features = pd.read_feather('../../data/processed/features_after_CNN.feather')
<<<<<<< HEAD
    data = pd.read_feather('../../data/processed/cifar_10_100_augmentation.feather')
=======
    data = pd.read_feather('../../data/processed/cifar_10_100.feather')
>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
    reshaped_image_array = new_image.reshape(1, 32, 32, 3)
    model = CNN.load_cnn_model('../saved_models/cnn_model_all_data.keras').model
    preprocessed_image = preprocess_input(reshaped_image_array)
    feature_layer_index = -2  # Index of the layer before the final dense layer
    feature_layer_output = model.layers[feature_layer_index].output
    # Create a Keras function to extract features from the given input images
    get_features = K.function([model.input], [feature_layer_output])
    new_image_features = get_features([preprocessed_image])[0]
    distances = []
    for row in data_features.values:
        distances.append(np.sum(np.abs(row - new_image_features)))
    sorted_indices = np.argsort(distances)
    sorted_data = data.iloc[sorted_indices]
    probabilities_indices = np.argsort(probabilities.reshape(-1))
    labels = probabilities_indices[-4:]  # Assuming this contains the labels you want
    filtered_data = sorted_data[sorted_data['label'].isin(labels)]
    four_closest_vectors = filtered_data.iloc[:4, 2:]
    similar_images_label = filtered_data.iloc[:4, 1].tolist()
    return four_closest_vectors


def encode_image(image_vector):
    # Reshape the image_vector to (32, 32, 3) if it's in the correct shape
    image_array = image_vector.reshape(3, 32, 32).transpose(1, 2, 0)
    # Convert the image_array (a NumPy array) to a PIL image
    img = Image.fromarray(image_array.astype('uint8'))
    # Convert the PIL image to a base64-encoded string
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()
    # plt.imshow(base64_image)
    # plt.show()
    return base64_image


def find_label_of_similar_image(image_index):
    if image_index is not None:
        image_label = similar_images_label[image_index]
        label = convert_to_category(image_label)
        return label
<<<<<<< HEAD
=======


def is_anomalous():
    return False

>>>>>>> 77badf082da1f688e173a30a10d8fb7b65deafae
