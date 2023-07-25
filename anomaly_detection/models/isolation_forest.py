from sklearn.ensemble import IsolationForest

train_images = []
test_images = []
flattened_train_images = train_images.reshape(train_images.shape[0], -1)
flattened_test_images = test_images.reshape(test_images.shape[0], -1)


isolation_forest = IsolationForest(contamination=0.25)
isolation_forest.fit(flattened_train_images)

anomaly_scores = isolation_forest.score_samples(flattened_test_images)

threshold = -0.5
for i in range(len(test_images)):
    if anomaly_scores[i] < threshold:
        print(f'anomaly detected in image {i}')

