import pandas as pd
import numpy as np
import pickle

path = '../data/features_6.feather'
features_df = pd.read_feather(path)
print('j')
# filter by TP

training_set = {}
validation_set = {}

grouped = features_df.groupby('label')

for class_key, class_df in grouped:
    twenty_percent = int(len(class_df) * 0.2)
    validation_indices = np.random.choice(class_df.index, size=twenty_percent, replace=False)

    validation_df = class_df.loc[validation_indices]
    validation_df = validation_df.iloc[:, 1:]

    validation_df.insert(0, 'belongs_to_class', 1)
    validation_set[class_key] = validation_df

    training_df = class_df.loc[~class_df.index.isin(validation_indices)]
    training_df = training_df.iloc[:, 1:]
    training_set[class_key] = training_df

for class_key, validation_df in validation_set.items():
    num_of_samples = len(validation_df)
    filtered_features = features_df[features_df['label'] != class_key]

    other_classes_df = filtered_features.sample(n=num_of_samples, random_state=42)
    other_classes_df = other_classes_df.iloc[:, 1:]

    other_classes_df.insert(0, 'belongs_to_class', 0)
    validation_set[class_key] = pd.concat([validation_df, other_classes_df], ignore_index=True)

with open('../data/training_set_6.pickle', 'wb') as file:
    pickle.dump(training_set, file)

with open('../data/validation_set_6.pickle', 'wb') as file:
    pickle.dump(validation_set, file)
