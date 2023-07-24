from classification_project.visualization.visualization import Visualization
from classification_project.models.cnn import CNN
from classification_project.preprocessing.preprocessing import Preprocessing
import pandas as pd

df = pd.read_csv('../../data/processed/cifar-10-100.csv')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)


model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras')
history = CNN.load_cnn_history('../saved_model/saved_cnn_history.pkl')

# Example usage:

#Visualization.plot_learning_curve(history)
#Visualization.plot_roc_curve(model.model, x_val, y_val)
#Visualization.plot_precision_recall_curve_multi_class(model.model, x_val, y_val)

accuracy, loss, f1 = Visualization.calculate_validation_metrics(model.model, x_val, y_val)

#Visualization.plot_convergence_graphs(history)

print("Validation Accuracy:", accuracy)
print("Validation Loss:", loss)
print("F1 Score:", f1)