import pandas as pd
import visualkeras
from classification_project.models.CNN1 import CNN
from classification_project.study.use_Visualization import plot_images_to_given_label
import pydot
import graphviz
from classification_project.models.CNN1 import CNN
# from classification_project.study.use_Visualization import plot_images_to_given_label
from keras.utils import plot_model
import keras

import albumentations

if __name__ == '__main__':
    # plot_images_to_given_label('people')
    # loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras').model
    # plot_model(loaded_model.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # loaded_model.model.summary()
    # input_shape = loaded_model.layers[0].input_shape
    # print("Input Shape:", input_shape)
    # visualkeras.layered_view(loaded_model.model)
    data = pd.read_csv('../../data/processed/cifar_10_100.csv')
    data.to_feather('../../data/processed/cifar_10_100.feather')
