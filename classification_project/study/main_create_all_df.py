from BL.data_handler_data_handler_cifar_10_100import DataHandlerCifar10Cifar100
from data_handler_cifar_10 import DataHandlerCifar10
from data_handler_cifar_100 import DataHandlerCifar100

if __name__ == '__main__':
    cifar10 = DataHandlerCifar10()
    cifar10.load_data_to_csv()

    cifar100 = DataHandlerCifar100()
    cifar100.load_data_to_csv()

    cifar_10_100_merged = DataHandlerCifar10Cifar100()
    cifar_10_100_merged.load_data_to_csv()
