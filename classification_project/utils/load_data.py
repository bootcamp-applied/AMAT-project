import pandas as pd
import numpy as np


def load_data():
    df = pd.read_csv('../../data/processed/cifar-10-100.csv')

    train = df[df['is_train'] == 1]
    test = df[df['is_train'] == 0]

    x_train = np.array(train.iloc[:,2:])
    x_test = np.array(test.iloc[:,2:])

    x_train = x_train.reshape(train.shape[0],3,32,32).transpose(0,2,3,1)
    x_test = x_test.reshape(test.shape[0],3,32,32).transpose(0,2,3,1)

    y_train = train['label'].values
    y_test = test['label'].values

    return x_train, y_train, x_test, y_test
