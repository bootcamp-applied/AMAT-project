import pandas as pd
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.cnn import CNN
from classification_project.models.CNN2 import CNN2

df = pd.read_csv('../../data/processed/cifar-10-100.csv')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras')
loaded_model_2 = CNN2.load_cnn_model('../saved_model/saved_cnn_model_2.keras')
loaded_history_model = CNN.load_cnn_history('../saved_model/saved_cnn_history.pkl')

accuracy = loaded_model.evaluate_accuracy(x_test,y_test)
accuracy_2= loaded_model_2.evaluate_accuracy(x_test,y_test)
print(f'accurcy 1: ${accuracy}')
print(f'accurcy 2: ${accuracy_2}')
