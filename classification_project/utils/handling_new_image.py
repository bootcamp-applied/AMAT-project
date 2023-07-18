import pandas as pd
import numpy as np
from visualization import Visualization
import matplotlib as plt
import os
import cv2
import csv
from ipywidgets import interact

class DataNewImage:

    def down_sampling_gaussian_filter(self, image, ksize = 5):
        image = cv2.GaussianBlur(image, (ksize, ksize), 0, 0)
        image = cv2.resize(image, (32, 32), fx=3, fy=3)
        return image

    def read_add_to_df(self,image_path,label):
        self.label = label
        # read image from path
        self.new_image = cv2.imread(image_path)
        # change from GBR to RGB
        self.new_image = cv2.cvtColor(self.new_image, cv2.COLOR_BGR2RGB)

        # transform photo's shape to Square
        shape = self.new_image.shape

        # If the shape of the image is different from a square

        if shape[0] != shape[1]:
            new_shape = min(shape[0], shape[1])
            self.new_image = self.new_image[:new_shape, :new_shape, :]

        shape = self.new_image.shape

        #  down-sampling if the new image is too big

        # if shape[0] > 32:
        #     origin_img = self.new_image
        #     self.new_image = self.down_sampling_gaussian_filter(self.new_image)
        #     small_img = self.new_image
        #     Visualization.show_downsampled_image(origin_img, small_img)

        # Up-sampling if the image is too small

        # if shape[0] < 32:
        #     origin_img = self.new_image
        #     self.new_image = cv2.resize(self.new_image, (32, 32), fx=3, fy=3, interpolation=cv2.INTER_CUBIC, ratio=(1, 10))
        #     big_img = self.new_image
        #     Visualization.show_downsampled_image(origin_img, big_img)
        origin_img = self.new_image
        self.new_image = cv2.resize(self.new_image, (32, 32), fx=3, fy=3)
        Visualization.show_downsampled_image(origin_img, self.new_image)

        # write to df
        with open('df','a', newline='') as file:
            writer = csv.writer(file)
            write_to_path= '../DAL/cifar_10_100_db.csv'
            # write the image to csv
            img_row = self.new_image.reshape(1,3072)
            df = pd.DataFrame(img_row)
            df.insert(0, 'is_train', 1)
            label = self.label
            df.insert(1, 'label', label)

            writer.writerows(df.values)