import pandas as pd
from classification_project.study.use_Visualization import Use_Visualization
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.cnn import CNN
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('../../data/processed/cifar-10-100-augmentation.csv')
    #preprocessing = Preprocessing(df)
   # preprocessing.prepare_data()
   # x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

   # model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras')
    #pareto to df
    Use_Visualization.pareto_to_df_label(df)
    #pareto to train_val_test
    Use_Visualization.pareto_tarin_val_test(df)
    #load_new_images()
    #y_pred= model.predict(x_test)
   # y_test = np.argmax(y_test, axis=1)
    #y_pred = np.argmax(y_pred, axis=1)
    #Use_Visualization.Confusion_matrix_cifar_10_100(y_test,y_pred)


