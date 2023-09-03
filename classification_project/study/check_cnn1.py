import pandas as pd
import os
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN1 import CNN1

df = pd.read_feather('../../data/processed/cifar_10_100_augmentation.feather')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

cnn_model = CNN1()
history = cnn_model.train(x_train, y_train, x_val, y_val)

accuracy = cnn_model.evaluate_accuracy(x_test, y_test)
print("Test accuracy:", accuracy)


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_all_data_trained_model.h5'

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
cnn_model.model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = cnn_model.model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])

print('Test accuracy:', scores[1])

print('Test accuracy:', scores[1])

