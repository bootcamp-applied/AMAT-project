# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
#
#
# class Preprocessing:
#     def __init__(self, data):
#         self.data = data
#
#     def normalize_data(self):
#         self.data.iloc[:, 2:] = self.data.iloc[:, 2:].astype('float32') / 255
#         # self.data.iloc[:, 2:] = self.data.iloc[:, 2:].values.reshape((-1, 32, 32, 3))
#         # self.data['label'] = to_categorical(self.data['label'])
#
#     def split_to_train_test_validation(self):
#         train = self.data[self.data['is_train'] == 1]
#         test = self.data[self.data['is_train'] == 0]
#         x_train = train.drop(['label', 'is_train'], axis=1)
#         x_train = x_train.values.reshape((-1, 32, 32, 3))
#         y_train = to_categorical(train['label'])
#         x_test = test.drop(['label', 'is_train'], axis=1)
#         x_test = x_test.values.reshape((-1, 32, 32, 3))
#         y_test = to_categorical(test['label'])
#         x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.3)
#         return x_train, y_train, x_val, y_val, x_test, y_test
#     def split_only(self):
#         train = self.data[self.data['is_train'] == 1]
#         test = self.data[self.data['is_train'] == 0]
#         x_train = train.drop(['label', 'is_train'], axis=1)
#         y_train = to_categorical(train['label'])
#         x_test = test.drop(['label', 'is_train'], axis=1)
#         y_test = to_categorical(test['label'])
#         x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.3)
#         return x_train, y_train, x_val, y_val, x_test, y_test
#
#     def cleaning_data(self):
#         self.data = self.data[~(self.data > 255).any(axis=1)]
#         print(self.data.shape)
#
#     def preparing_data(self):
#         self.cleaning_data()
#         self.normalize_data()

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class Preprocessing:

    def __init__(self, data):
        self.data = data

    def split_data(self, include_validation=True, one_hot_encoder=False):
        train = self.data[self.data['is_train'] == 1]
        test = self.data[self.data['is_train'] == 0]

        # important change!!!!
        x_train = train.drop(['label', 'is_train'], axis=1).values.reshape((-1, 3, 32, 32)).transpose(0,2,3,1) # ((-1, 32, 32, 3))
        y_train = train['label'].values

        x_test = test.drop(['label', 'is_train'], axis=1).values.reshape((-1, 3, 32, 32)).transpose(0,2,3,1) # ((-1, 32, 32, 3))
        y_test = test['label'].values

        if one_hot_encoder:
            y_train = to_categorical(train['label'])
            y_test = to_categorical(test['label'])

        if include_validation:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)
            return x_train, y_train, x_val, y_val, x_test, y_test
        else:
            return x_train, y_train, x_test, y_test

    def normalize_data(self):
        self.data.iloc[:, 2:] = self.data.iloc[:, 2:].astype('float32') / 255

    def clean_data(self):
        self.data = self.data[~(self.data > 255).any(axis=1)]


    def prepare_data(self):
        self.clean_data()

        self.normalize_data()

