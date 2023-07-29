import pandas as pd
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN
from visu_images import Visualization

df = pd.read_csv('../../data/processed/cifar-10-100-argumentation.csv')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model.kears')#kears
loaded_history_model = CNN.load_cnn_history('../saved_model/saved_cnn_history.pkl')

Visualization().plot_correct_and_incorrect_classified_images(x_test, y_test, loaded_model)
