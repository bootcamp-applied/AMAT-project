import pandas as pd
import numpy as np
from classification_project.visualization.visualization import Visualization
import os
import cv2
import matplotlib.pyplot as plt
import csv
# from ipywidgets import interact

class NewImage:

    def __init__(self):
        self.image = None
        self.shape = None
        self.label = None
        self.origin_img = None

    def down_sampling_gaussian_filter(self, image, ksize=5):
        image = cv2.GaussianBlur(image, (ksize, ksize), 0, 0)
        image = cv2.resize(image, (32, 32), fx=3, fy=3)
        return image

    def image_handle(self, image):
        # transform photo's shape to Square
        # If the shape of the image is different from a square
        if self.shape[0] != self.shape[1]:
            new_shape = min(self.shape[0], self.shape[1])
            cut = int((max(self.shape[0], self.shape[1]) - new_shape)/2)
            end = int((max(self.shape[0], self.shape[1]))-cut)
            if self.shape[0] < self.shape[1]:
                new_image = image[:, cut:end, :]
            else:
                new_image = image[cut:end, :, :]

        # shape = self.new_image.shape
        self.origin_img = new_image
        new_image = cv2.resize(new_image, (32, 32), fx=3, fy=3)
        return new_image

    def add_image_to_csv(self, image, label):
        self.image = image
        self.shape = self.image.shape
        self.label = label
        new_image = self.image_handle(self.image)
        # change from GBR to RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        Visualization.show_downsampled_image(self.origin_img, new_image)
        write_to_path = '../../data/processed/cifar-10-100.csv'
        with open(write_to_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # write the image to csv
            img_row = new_image.transpose(2, 0, 1).reshape(1, -1)
            df = pd.DataFrame(img_row)
            df.insert(0, 'is_train', 1)
            df.insert(1, 'label', self.label)
            writer.writerows(df.values)



