import pickle
import pandas as pd
import os.path
import csv


class DataHandlerCifar10:
    def unpickele(self, file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='latin1')
        return data_dict

    def convert_dict_to_df(self, data, is_train):
        images = data['data']
        images_reshaped = images.reshape(images.shape[0], -1)
        df = pd.DataFrame(images_reshaped)
        df.insert(0, 'is_train', is_train)
        labels = data['labels']
        df.insert(1, 'label', labels)
        return df
    def load_data_to_csv(self):
        train_files = r'..\datasets\cifar-10-batches-py\data_batch_'
        test_file = r'..\datasets\cifar-10-batches-py\test_batch'
        write_to_path = r'..\DAL\cifar-10-df.csv'
        if os.path.exists(write_to_path):
            os.remove(write_to_path)
        with open(write_to_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for i in range(5):
                # convert the file to a dictionary
                data_train_dict = self.unpickele(train_files+str(i+1))
                # convert the dictionary to a dataframe
                # df = pd.DataFrame(data={'data': data['data'].tolist(), 'label': data['labels'], 'is_train': 1})
                df = self.convert_dict_to_df(data_train_dict, 1)
                # write the data to csv
                if os.path.isfile(write_to_path) and os.path.getsize(write_to_path) == 0:
                    df.to_csv(write_to_path, index=False)
                else:
                    writer.writerows(df.values)

            data_test_dict = self.unpickele(test_file)
            df = self.convert_dict_to_df(data_test_dict, 1)
            # df = pd.DataFrame(data={'data': data['data'].tolist(), 'label': data['labels'], 'is_train': 0})
            writer.writerows(df.values)