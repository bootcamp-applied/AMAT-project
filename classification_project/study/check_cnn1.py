# import pandas as pd
# from classification_project.preprocessing.preprocessing import Preprocessing
# from classification_project.models.CNN1 import CNN
#
# df = pd.read_csv('../../data/processed/cifar-10-100-augmentation.csv')
# preprocessing = Preprocessing(df)
# preprocessing.prepare_data()
# x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)
#
# cnn_model = CNN()
# history = cnn_model.train(x_train, y_train, x_val, y_val)
#
# model_filename = '../saved_model/cnn_model_1.h5'
# cnn_model.save_model(model_filename)
#
# # Load the training history from the file
# history_filename = '../saved_model/cnn_history_1.joblib'
# cnn_model.save_history(history_filename)
#
# accuracy = cnn_model.evaluate_accuracy(x_test, y_test)
# print("Test accuracy:", accuracy)
#
import pandas as pd
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN

df = pd.read_csv('data/processed/cifar-10-100-augmentation.csv')
print(df.shape)
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

cnn_model = CNN()
history = cnn_model.train(x_train, y_train, x_val, y_val)

model_filename = 'classification_project/saved_model/cnn_model_all_data.keras'
cnn_model.save_model(model_filename)

# Load the training history from the file
history_filename = 'classification_project/saved_model/cnn_history_all_data.pkl'
cnn_model.save_history(history_filename)

accuracy = cnn_model.evaluate_accuracy(x_test, y_test)
print("Test accuracy:", accuracy)

