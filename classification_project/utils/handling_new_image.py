import pandas as pd
import numpy as np
from classification_project.visualization.visualization import Visualization
import os
import cv2
import matplotlib.pyplot as plt
import csv
#from ipywidgets import interact

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
            cut = int((max(shape[0], shape[1]) - new_shape)/2)
            end = int((max(shape[0], shape[1]))-cut)
            if(shape[0] < shape[1]):
                self.new_image = self.new_image[:, cut:end, :]
            else:
                self.new_image = self.new_image[cut:end, :, :]

        shape = self.new_image.shape


        origin_img = self.new_image
        self.new_image = cv2.resize(self.new_image, (32, 32), fx=3, fy=3)
        Visualization.show_downsampled_image(origin_img, self.new_image)

        # write to df
        write_to_path = '../../data/processed/cifar-10-100.csv'
        with open(write_to_path,'a', newline='') as file:
            writer = csv.writer(file)
            # write the image to csv

            img_row = self.new_image.transpose(2, 0, 1).reshape(1, -1)

            df = pd.DataFrame(img_row)
            df.insert(0, 'is_train', 1)
            label = self.label
            df.insert(1, 'label', label)
            writer.writerows(df.values)