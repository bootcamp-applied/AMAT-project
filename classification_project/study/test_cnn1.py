import pandas as pd
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN1

df = pd.read_csv('../../data/processed/cifar-10-100.csv')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

cnn_model = CNN1()
history = cnn_model.train(x_train, y_train, x_val, y_val)

accuracy = cnn_model.evaluate_accuracy(x_test, y_test)
print("Test accuracy:", accuracy)


import os

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_model_1.h5'

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
cnn_model.model.save(model_path)
print('Saved trained model at %s ' % model_path)


# model_filename = '../saved_models/cnn_model_1'
# cnn_model.save_model(model_filename)
#
# # Load the training history from the file
# history_filename = '../saved_model/cnn_history_1.joblib'
# cnn_model.save_history(history_filename)
