import pandas as pd
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def convert_dict_to_df(data, is_train):
    images = data['data']
    images_reshaped = images.reshape(images.shape[0], -1)
    df = pd.DataFrame(images_reshaped)
    df.insert(0, 'is_train', is_train)
    labels = data['coarse_labels']
    df.insert(1,'label',labels)
    return df

class DataHandlerCifar100:

    def load_data_to_csv(self):

        train_file = '../../data/raw/cifar-100-python/train'
        test_file = '../../data/raw/cifar-100-python/test'

        train_data = unpickle(train_file)
        test_data = unpickle(test_file)

        train_df = convert_dict_to_df(train_data, 1)
        test_df = convert_dict_to_df(test_data, 0)
        df = pd.concat([train_df, test_df], ignore_index=True)

        path = '../../data/processed/cifar_100.csv'

        if os.path.exists(path):
            # If the file exists, delete it
            os.remove(path)

        df.to_csv(path, index=False, encoding='utf-8', mode='w')
