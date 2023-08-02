import numpy as np
import joblib
import tensorflow as tf
from anomaly_detection.utils.download_image_and_rezise import download_and_resize_images
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

features_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

label_dict = {
  "0": "airplane",
  "1": "automobile",
  "2": "bird",
  "3": "cat",
  "4": "deer",
  "5": "dog",
  "6": "frog",
  "7": "horse",
  "8": "ship",
  "9": "truck",
  "10": "fish",
  "11": "people",
  "12": "flowers",
  "13": "trees",
  "14": "fruit and vegetables"
}

urls = [
    "https://images.pexels.com/photos/17533716/pexels-photo-17533716/free-photo-of-royal-enfield.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/17541729/pexels-photo-17541729/free-photo-of-sunset-beach.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/1046493/pexels-photo-1046493.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/17541735/pexels-photo-17541735/free-photo-of-beach.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/17541726/pexels-photo-17541726/free-photo-of-jeep.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/17353009/pexels-photo-17353009/free-photo-of-royal-enfield-bike.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/1883385/pexels-photo-1883385.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/1624076/pexels-photo-1624076.jpeg?auto=compress&cs=tinysrgb&w=1600",
    "https://images.pexels.com/photos/60090/pexels-photo-60090.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    "https://images.pexels.com/photos/1145274/pexels-photo-1145274.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
]

test_images = download_and_resize_images(urls, resize_shape=(32, 32))
# or test_images = df.iloc[:9,2:]
test_images = np.array(test_images)
# test_images = test_images.reshape(test_images.shape[0],3,32,32)
# test_images = test_images.transpose(0,2,3,1)
test_images = tf.image.resize(test_images, (96, 96))
# for examplt plt.imshow(np.array(test_images[7]).astype(int))
features = features_extractor.predict(preprocess_input(test_images))
flatten_features = features.reshape(features.shape[0], -1)

anomaly_scores = {}

models = joblib.load("../saved_models/isolation_forest_models_low_contamination.joblib")

for k,model in models.items():
  score = model.score_samples(flatten_features)
  anomaly_scores[label_dict[str(k)]] = score


predictions = {}
for k,model in models.items():
  predict = model.predict(flatten_features)
  predictions[label_dict[str(k)]] = predict

{key: values_list[2] for key, values_list in predictions.items()}
