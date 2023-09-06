from classification_project.data_handler.data_handler_cifar_10 import DataHandlerCifar10
from classification_project.data_handler.data_handler_cifar_100 import DataHandlerCifar100
from classification_project.data_handler.data_handler_cifar_10_100 import DataHandlerCifar10Cifar100
from classification_project.data_handler.data_handler_cifar_10_100_augmentation import DataHandlerCifar10Cifar100Augmentation
from classification_project.data_handler.data_handler_all_data_augmentation import DataHandlerCifar10Cifar100AllAugmentation
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.visualization.visualization import Visualization
import classification_project.utils.add_images

if __name__ == '__main__':
    # cifar10
    cifar10 = DataHandlerCifar10()
    cifar10.load_data_to_csv()
    # cifar100
    cifar100 = DataHandlerCifar100()
    cifar100.load_data_to_csv()
    # cifar10 with cifar100
    cifar10_100=DataHandlerCifar10Cifar100()
    cifar10_100.load_data_to_csv()
    # augmentation
    df_10_100_augmentation = DataHandlerCifar10Cifar100Augmentation()
    df_10_100_augmentation.load_data_to_csv()
    # all augmentation
    df_all = DataHandlerCifar10Cifar100AllAugmentation()
    df_all.load_data_to_csv()
    # add new images
    # from where????
    # useDataNewImage()









