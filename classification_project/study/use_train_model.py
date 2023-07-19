import pandas as pd
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.cnn import CNN

df = pd.read_csv('../../data/processed/cifar-10.csv')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras')
y_pred = loaded_model.predict(x_test)
loaded_history_model = CNN.load_cnn_history('../saved_model/saved_cnn_history.pkl')

accuracy = loaded_history_model.evaluate_accuracy(x_test, y_test)
print("Test accuracy:", accuracy)

