from ..visualization.visualization import Visualization
from ..preprocessing.preprocessing import Preprocessing

import pandas as pd
import numpy as np

class Use_Visualization:
    @staticmethod
    def pareto_tarin_val_test(df):
        data= Preprocessing(df)
        x_train, y_train, x_val, y_val, x_test, y_test=data.split_data()
        y_train_count = y_train.shape[0]
        y_val_count = y_val.shape[0]
        y_test_count = y_test.shape[0]

        # Combine the counts into a single list
        y = []
        for count in range(y_train_count):
            y.append(0)
        for count in range(y_val_count):
            y.append(1)
        for count in range(y_test_count):
            y.append(2)

        dict={
            '0':'train',
            '1':'val',
            '2':'test'
        }
        # Visualization using Pareto (assuming you have imported the necessary module)
        Visualization.Pareto(y,dict)

    @staticmethod
    def pareto_to_df_label(df):
        Visualization.Pareto(df['label'])