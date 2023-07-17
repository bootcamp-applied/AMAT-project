import pandas as pd
import numpy as np
from visualization import Visualization
from preprocessing import PreProcessing

if __name__ == '__main__':
    # Read the CSV file into a DataFrame
    df = pd.read_csv('../DAL/cifar_10_100_db.csv')
    #Visualization.Pareto(df['label'])
    x = df.iloc[:, 2:]
    # y = df['label']
    # Visualization.TSNE(x, y)

    #Visualization.Pareto(y)
    data = PreProcessing(df)
    x_train, y_train, x_val, y_val, x_test, y_test = data.split_only()
    # y_train = y_train.flatten()
    # y_val= y_val.flatten()
    # y_test=y_test.flatten()
    # pd.Series(y_train)
    # pd.Series(y_val)
    # pd.Series(y_test)
    # y_train_count=y_train.value_counts()
    # y_val_count=y_val.value_counts()
    # y_test_count=y_test.value_counts()
    # y_train_count = pd.Series(np.unique(y_train, return_counts=True)[1])
    # y_val_count = pd.Series(np.unique(y_val, return_counts=True)[1])
    # y_test_count = pd.Series(np.unique(y_test, return_counts=True)[1])
    # y=[]
    # for i in range(y_train_count):
    #     y.push(0)
    # for i in range(y_val_count):
    #     y.push(1)
    # for i in range(y_test_count):
    #     y.push(2)
    print(y_test.shape)
    print(y_val.shape)
    print(y_train.shape)
    y_train_count = pd.Series(np.unique(y_train, return_counts=True)[0])
    y_val_count = pd.Series(np.unique(y_val, return_counts=True)[0])
    y_test_count = pd.Series(np.unique(y_test, return_counts=True)[0])
    # Combine the counts into a single list
    y = []
    for count in y_train_count.values:
        y.append(0)
    for count in y_val_count.values:
        y.append(1)
    for count in y_test_count.values:
        y.append(2)
    # Visualization using Pareto (assuming you have imported the necessary module)
    Visualization.Pareto(y)
