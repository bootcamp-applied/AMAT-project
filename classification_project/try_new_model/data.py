import pandas as pd
import random
import os
# read data of cifar100

class DataToOtherModel:
    @staticmethod
    def prepare():
        df = pd.read_csv('../../data/processed/cifar_100.csv')
        print("aaa")
        print(df.head())
        # remain only the other class
        # delete all existed superclass 1,2,4,14,17,19

        df = df[(df['label'] != 1) & (df['label'] != 2) & (df['label'] != 4) & (df['label'] != 14) & (df['label'] != 17) & (df['label'] != 18)]
        print("bbb")
        # 5000 train 1000 test
        # 14 new super classes
        # 429 from each super class drop 6
        df = df.groupby('label').head(429)
        print(len(df))
        index_to_delete = random.sample(range(len(df)), 6)
        print(index_to_delete)
        print(type(index_to_delete))
        df = df.loc[~df.index.isin(index_to_delete)]
        df.reset_index(drop=True, inplace=True)
        df['label'] = 15

        num_rows = len(df)
        is_train_values = [1] * 5000 + [0] * (num_rows - 5000)
        random.shuffle(is_train_values)
        df['is_train'] = is_train_values

        # combine data with the normal data in the base model
        dff_without_df = pd.read_feather('../../data/processed/cifar_10_100_augmentation.feather')
        df = pd.concat([df, dff_without_df], ignore_index=True)

        path = '../../data/processed/cifar_10_100_all.feather'
        if os.path.exists(path):
            os.remove(path)

        df.to_feather(path)
