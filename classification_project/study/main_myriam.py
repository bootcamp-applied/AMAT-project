import pandas as pd

from BL.preprocessing import PreProcessing
from data_handler_cifar_10 import DataHandlerCifar10
from data_handler_cifar_100 import DataHandlerCifar100
from data_handler_data_handler_cifar_10_100import DataHandlerCifar10Cifar100
from preprocessing import PreProcessing
from visualization import Visualization
import pandas as pd

if __name__ == '__main__':
    cifar10 = DataHandlerCifar10()
    cifar10.load_data_to_csv()
    cifar100 = DataHandlerCifar100()
    cifar100.load_data_to_csv()
    cifar10_100=DataHandlerCifar10Cifar100()
    cifar10_100.load_data_to_csv()
    data = pd.read_csv('../DAL/cifar_10_100_db.csv')
    data = PreProcessing(data)
    data.preparing_data()
    x_train, y_train, x_val, y_val, x_test, y_test = data.split_to_train_test_validation()
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
