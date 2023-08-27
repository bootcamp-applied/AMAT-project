import csv
from ..utils.add_imgs_cifar_100 import plot_images
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from ..visualization.visualization import Visualization
from ..preprocessing.preprocessing import Preprocessing
import json
import os

class Use_Visualization:
    @staticmethod
    def pareto_tarin_val_test(df):
        data= Preprocessing(df)
        x_train, y_train, x_val, y_val, x_test, y_test=data.split_data()
        y_train_count = y_train.shape[0]
        y_val_count = y_val.shape[0]
        y_test_count = y_test.shape[0]

        # Combine the counts into a single list
        y = []
        for count in range(y_train_count):
            y.append(0)
        for count in range(y_val_count):
            y.append(1)
        for count in range(y_test_count):
            y.append(2)

        dict={
            '0':'train',
            '1':'val',
            '2':'test'
        }
        # Visualization using Pareto (assuming you have imported the necessary module)
        Visualization.Pareto(y,dict)

    @staticmethod
    def pareto_to_df_label(df):
        file_path = os.path.join("classification_project", "utils", "dict.json")
        with open('../utils/dict.json') as f:
            data = json.load(f)
        Visualization.Pareto(df['label'],data)

    @staticmethod
    def Confusion_matrix_cifar_10_100(y_true,y_pred):
        with open('../utils/dict.json') as f:
            data = json.load(f)
        class_names = list(data.values())
        Visualization.Confusion_matrix(y_true,y_pred,class_names)


def plot_images_to_given_label(label):
    map_label = '../utils/dict.json'
    with open(map_label, 'r') as f:
        label_dict = json.load(f)
    label_key = next((key for key, val in label_dict.items() if val == label), None)
    csv_file_path = '../../data/processed/cifar_10_100.csv'
    # selected_r = data[data['label'] == str(label_key)]
    rows_to_read = 9
    selected_rows = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['label'] == str(label_key):
                row_values = list(row.values())
                selected_rows.append(row_values[2:])
                if len(selected_rows) == rows_to_read:
                    break
    images_to_plot = []
    for image_id in range(len(selected_rows)):
        image = np.array(selected_rows[image_id], dtype=np.int32)
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        images_to_plot.append(image)

    plot_images(images_to_plot)
