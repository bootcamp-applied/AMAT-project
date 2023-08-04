import pandas as pd
import os

class DataHandlerCifar10Cifar100AllAugmentation:
    def __init__(self):
        self.cifar_10_100_db = None
        self.df_augmentation = None

    def read_from_csv(self):
        path_augmentation = '../../data/processed/all_augmentation.csv'
        path_cifar_10_100='../../data/processed/cifar-10-100-augmentation.csv'
        df_cifar_10_100=pd.read_csv(path_cifar_10_100)
        df_augmentation = pd.read_csv(path_augmentation)
        return df_cifar_10_100,df_augmentation

    # without sub class
    def load_data_to_csv(self):
        self.df_cifar_10_100, self.df_augmentation = self.read_from_csv()

        # new labels
        #self.df_augmentation.label = self.df_augmentation['label'].map({1: 10, 14: 11, 2: 12, 17: 13, 4: 14})

        merged_df = pd.concat([self.df_cifar_10_100, self.df_augmentation])
        path = '../../data/processed/cifar-10-100-all-data-augmentation.csv'

        if os.path.exists(path):
            # If the file exists, delete it
            os.remove(path)

        merged_df.to_csv(path, index=False, encoding='utf-8', mode='w')
