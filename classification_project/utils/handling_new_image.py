import pandas as pd
import numpy as np

from classification_project.utils.add_imgs_cifar_100 import plot_images
from classification_project.visualization.visualization import Visualization
import os
import cv2
import csv
#from classification_project.utils.add_imgs_cifar_100 import plot_images
import matplotlib.pyplot as plt


def down_sampling_gaussian_filter(image):
    origin_img = image
    image = cv2.GaussianBlur(image, (5, 5), 0, 0)
    Visualization.show_downsampled_image(origin_img, image)
    image = cv2.resize(image, (32, 32), fx=3, fy=3)
    return image

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
        new_image = image
        shape = image.shape
        if shape[0] != shape[1]:
            new_shape = min(shape[0], shape[1])
            cut = int((max(shape[0], shape[1]) - new_shape)/2)
            end = int((max(shape[0], shape[1]))-cut)
            if shape[0] < shape[1]:
                new_image = image[:, cut:end, :]
            else:
                new_image = image[cut:end, :, :]


        # shape = self.new_image.shape
        self.origin_img = new_image
        new_image = down_sampling_gaussian_filter(new_image)
        return new_image

        # shape = new_image.shape
        origin_img = new_image
        new_image = cv2.resize(new_image, (32, 32), fx=3, fy=3)
        return new_image, origin_img


    def add_image_to_csv(self, image, label):
        image = image
        shape = image.shape
        label = label
        new_image, origin_img = self.image_handle(image)
        # change from GBR to RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

        self.origin_img = cv2.cvtColor(self.origin_img, cv2.COLOR_BGR2RGB)
        Visualization.show_downsampled_image(self.origin_img, new_image)
        write_to_path = '../../data/processed/cifar-10-100.csv'

        Visualization.show_downsampled_image(origin_img, new_image)
        write_to_path = '../../data/processed/cifar_10_100.csv'

        with open(write_to_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # write the image to csv
            img_row = new_image.transpose(2, 0, 1).reshape(1, -1)
            df = pd.DataFrame(img_row)
            df.insert(0, 'is_train', 1)
            df.insert(1, 'label', label)
            writer.writerows(df.values)

    def test_new_images(self):
        df = pd.read_csv('../../data/processed/cifar-10-100.csv')
        last_img = df.iloc[-1, :][2:].values.reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(last_img)
        plt.show()


    def test_new_images(self):
        df = pd.read_csv('../../data/processed/cifar_10_100.csv')
        last_imgs = df.iloc[-9, :][2:].values.reshape(3, 32, 32).transpose(1, 2, 0)
        plot_images(last_imgs)
