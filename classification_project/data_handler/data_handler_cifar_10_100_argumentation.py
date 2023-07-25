import pandas as pd
import os

class DataHandlerCifar10Cifar100Argumentation:
    def __init__(self):
        self.cifar_10_100_db = None
        self.df_argumentation = None

    def read_from_csv(self):
        path_argumentation = '../../data/processed/argumentation.csv'
        path_cifar_10_100='../../data/processed/cifar-10-100.csv'
        df_cifar_10_100=pd.read_csv(path_cifar_10_100)
        df_argumentation = pd.read_csv(path_argumentation)
        return df_cifar_10_100,df_argumentation

    # without sub class
    def load_data_to_csv(self):
        self.df_cifar_10_100, self.df_argumentation = self.read_from_csv()

        merged_df = pd.concat([self.df_cifar_10_100, self.df_argumentation])
        path = '../../data/processed/cifar-10-100-argumentation.csv'

        if os.path.exists(path):
            # If the file exists, delete it
            os.remove(path)

        merged_df.to_csv(path, index=False, encoding='utf-8', mode='w')
