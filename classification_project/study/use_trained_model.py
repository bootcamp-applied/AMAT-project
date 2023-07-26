import pandas as pd
import numpy as np
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.cnn import CNN
from classification_project.models.CNN2 import CNN2
from classification_project.visualization.visualization import Visualization
from classification_project.study.use_Visualization import Use_Visualization

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


Visualization.plot_roc_curve(loaded_model.model, x_val, y_val)
Visualization.plot_precision_recall_curve_multi_class(loaded_model.model, x_val, y_val)

y_pred= loaded_model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
Use_Visualization.Confusion_matrix_cifar_10_100(y_test,y_pred)

Visualization.plot_convergence_graphs(loaded_history_model)
accuracy, loss, f1 = Visualization.calculate_validation_metrics(loaded_model.model, x_val, y_val)

print("Validation Accuracy:", accuracy)
print("Validation Loss:", loss)
print("F1 Score:", f1)