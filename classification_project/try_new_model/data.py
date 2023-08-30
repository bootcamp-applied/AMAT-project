import pandas as pd
import random
import os
# read data of cifar100

class DataToOtherModel:
    @staticmethod
    def prepare(self):
        df = pd.read_csv('../../data/processed/cifar-100.csv')
        print(df.head())
        # remain only the other class
        # delete all existed superclass 1,2,4,14,17,19

        df = df[(df['lable'] != 1) & (df['lable'] != 2)& (df['lable'] != 4) & (df['lable'] != 14) & (df['lable'] != 17) & (df['lable'] != 18)]

        # 5000 train 1000 test
        # 74 new classes
        # 429 from each super class drop 6
        df = df.groupby('lable').head(429)
        index_to_delete = random.sample(range(len(df)), 6)
        df.drop(index_to_delete, inplace=True)
        df['Label'] = 15

        num_rows = len(df)
        is_train_values = [1] * 5000 + [0] * (num_rows - 5000)
        random.shuffle(is_train_values)
        df['is_train'] = is_train_values

        # combine data with the normal data in the base model
        dff_without_df = pd.read_csv('../../data/processed/cifar_10_100_augmentation.csv')
        df = pd.concat([df, dff_without_df], ignore_index=True)

        path = '../../data/processed/cifar_10_100_all.feather'
        if os.path.exists(path):
            os.remove(path)

        df.to_feather(path)
