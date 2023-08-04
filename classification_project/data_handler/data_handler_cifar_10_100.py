import pandas as pd
import os

class DataHandlerCifar10Cifar100:
    def __init__(self):
        self.cifar_10_db = None
        self.cifar_100_db = None

    def read_from_csv(self):
        path_cifar_10 = '../../data/processed/cifar_10.csv'
        path_cifar_100 = '../../data/processed/cifar_100.csv'
        df_cifar_10 = pd.read_csv(path_cifar_10)
        df_cifar_100 = pd.read_csv(path_cifar_100)
        return df_cifar_10, df_cifar_100

    # without sub class
    def load_data_to_csv(self):
        self.cifar_10_db, self.cifar_100_db = self.read_from_csv()
        # cifar 100 super class
        self.cifar_100_db = self.cifar_100_db[
            (self.cifar_100_db.label == 1) | (self.cifar_100_db.label ==14) | (
                        self.cifar_100_db.label == 2) | (self.cifar_100_db.label == 17) |
                        (self.cifar_100_db.label == 4)]
        #new labels
        self.cifar_100_db.label = self.cifar_100_db['label'].map({1: 10, 14: 11, 2:12, 17:13, 4:14})

        # concat cifar 10 with cifar 100
        merged_df = pd.concat([self.cifar_10_db, self.cifar_100_db])
        path = '../../data/processed/cifar_10_100.csv'

        if os.path.exists(path):
            # If the file exists, delete it
            os.remove(path)

        merged_df.to_csv(path, index=False, encoding='utf-8', mode='w')
