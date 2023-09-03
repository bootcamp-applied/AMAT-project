import pickle
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

with open('../data/training_set_6.pickle', 'rb') as file:
    training_set = pickle.load(file)

models = {}

for class_key, class_df in training_set.items():
    f = np.array(class_df)
    flatten_features = f.reshape(f.shape[0], -1)
    isolation_forest = IsolationForest(contamination=0.0000001, random_state=42)
    isolation_forest.fit(flatten_features)
    models[class_key] = isolation_forest



models_filename = "../saved_models/isolation_forest_models_6.joblib"
joblib.dump(models, models_filename)
