# note that the positive class represents the class of the anomalies

import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

features_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

with open('../data/validation_set.pickle', 'rb') as file:
    validation_set = pickle.load(file)

class_df = validation_set[0]
x_positive = class_df[class_df['belongs_to_class'] == 0].drop('belongs_to_class', axis=1)
x_negative = class_df[class_df['belongs_to_class'] == 1].drop('belongs_to_class', axis=1)

x_positive = np.array(x_positive)
x_positive = x_positive.reshape(len(x_positive), 3, 32, 32)
x_positive = x_positive.transpose(0, 2, 3, 1)
x_positive = tf.image.resize(x_positive, (96, 96))
x_positive_features = features_extractor.predict(preprocess_input(x_positive))
x_positive_flatten_features = x_positive_features.reshape(x_positive_features.shape[0], -1)
















# from sklearn.model_selection import train_test_split
#
# airplain_df = df[df['label']==0]
#
# airplain_images = airplain_df.iloc[:,2:]
# train_images, val_images = train_test_split(airplain_images, test_size=0.2, random_state=42)
# num_of_val = val_images.shape[0]
# df_without_airplane = df[df['label']!=0].sample(n=num_of_val, random_state=42)
# images_without_airplane = df_without_airplane.iloc[:,2:]
# val_images['is_airplane'] = 1
# images_without_airplane['is_airplane'] = 0
# new_df = pd.concat([val_images, images_without_airplane], ignore_index=True)
#
# df_from_feather = pd.read_feather('../data/df.feather')


