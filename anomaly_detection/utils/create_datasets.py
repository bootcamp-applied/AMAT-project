import pandas as pd
import numpy as np
import pickle

# I'm going to build training set and validation set

df = pd.read_feather('../data/data.feather')

training_set = {}
validation_set = {}

grouped = df.groupby('label')

for class_key, class_df in grouped:
    twenty_percent = int(len(class_df) * 0.2)
    validation_indices = np.random.choice(class_df.index, size=twenty_percent, replace=False)
    validation_df = class_df.loc[validation_indices]
    validation_df = validation_df.iloc[:, 2:]
    validation_df.insert(0, 'belongs_to_class', 1)
    validation_set[class_key] = validation_df

    training_df = class_df.loc[~class_df.index.isin(validation_indices)]
    training_df = training_df.iloc[:, 2:]
    training_set[class_key] = training_df

for class_key, validation_df in validation_set.items():
    num_of_samples = len(validation_df)
    filtered_df = df[df['label'] != class_key]
    other_classes_df = filtered_df.sample(n=num_of_samples, random_state=42)
    # or - other_classes_df = df[df['label'] != class_key].sample(n=num_of_samples, random_state=42)
    other_classes_df = other_classes_df.iloc[:, 2:]
    other_classes_df.insert(0, 'belongs_to_class', 0)
    validation_set[class_key] = pd.concat([validation_df, other_classes_df], ignore_index=True)

with open('../data/training_set.pickle', 'wb') as file:
    pickle.dump(training_set, file)

with open('../data/validation_set.pickle', 'wb') as file:
    pickle.dump(validation_set, file)






# df_with_split = df.copy()
# df_with_split['is_val'] = 0
#
# def set_val(group):
#     num_rows = group.shape[0]
#     num_rows_to_set_one = max(1, int(num_rows * 0.2))  # 20% of rows, at least 1 row
#     group_copy = group.copy()  # Create a copy of the group
#     group_copy.loc[group_copy.index[:num_rows_to_set_one], 'is_val'] = 1
#     return group_copy
#
# df_with_split.groupby('label').apply(set_val)

# df_with_split.to_feather('../data/data_with_split.feather')
