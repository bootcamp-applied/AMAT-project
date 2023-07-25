from classification_project.data_handler.data_handler_cifar_10_100 import DataHandlerCifar10Cifar100
from classification_project.data_handler.data_handler_cifar_10 import DataHandlerCifar10
from classification_project.data_handler.data_handler_cifar_100 import DataHandlerCifar100
from classification_project.data_handler.data_handler_cifar_10_100_argumentation import DataHandlerCifar10Cifar100Argumentation

if __name__ == '__main__':
    # cifar10 = DataHandlerCifar10()
    # cifar10.load_data_to_csv()
    #
    # cifar100 = DataHandlerCifar100()
    # cifar100.load_data_to_csv()
    #
    # cifar_10_100_merged = DataHandlerCifar10Cifar100()
    # cifar_10_100_merged.load_data_to_csv()

    df_10_100_argumentation=DataHandlerCifar10Cifar100Argumentation()
    df_10_100_argumentation.load_data_to_csv()