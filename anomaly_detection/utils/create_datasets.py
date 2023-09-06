import pickle
import pandas as pd
import numpy as np


# Load the CNN model
cnn_model = load_model('../../classification_project/saved_models/cnn_model_all_data.keras')
# cnn_model = load_model('../../classification_project/saved_models/best_model.h5')

# Create a feature extractor model using a specific layer
feat_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('dense_1').output)

# Read the data from the feather file
df = pd.read_feather('../data/df.feather')

images = np.array(df.iloc[:, 2:]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Normalize the images
normalized_images = images.astype('float32') / 255

# Extract features using the feature extractor model
features = feat_extractor.predict(normalized_images)

# Reshape the features for saving to a DataFrame
flatten_features = features.reshape(features.shape[0], -1)

# Create a DataFrame with the flattened features and labels
features_df = pd.DataFrame(flatten_features)
features_df.insert(0, 'label', df['label'])

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
