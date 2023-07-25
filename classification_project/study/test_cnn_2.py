import pandas as pd
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.CNN2 import CNN2

df = pd.read_csv('../../data/processed/cifar-10-100.csv')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

# loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model_2.h5')
# loaded_history_model = CNN.load_cnn_history('../saved_model/saved_cnn_history_2.pkl')

cnn_model = CNN2(num_classes=15)
history = cnn_model.train(x_train, y_train, x_val, y_val)

cnn_model.save_model('../saved_model/saved_cnn_model_2.keras')
cnn_model.save_history('../saved_model/saved_cnn_model_2.pkl')

accuracy = cnn_model.evaluate_accuracy(x_test, y_test)
print("Test accuracy:", accuracy)

