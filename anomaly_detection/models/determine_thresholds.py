# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc, confusion_matrix
# import pickle
# import seaborn as sns
#
# with open('../data/validation_set_6.pickle', 'rb') as file:
#     validation_set = pickle.load(file)
#
# models = joblib.load("../saved_models/isolation_forest_models_6.joblib")
#
# class_names = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck",
#     "fish", "people", "flowers", "trees", "fruit and vegetables"
# ]
#
# from sklearn.metrics import confusion_matrix
#
# plt.figure(figsize=(15, 10))
# for class_idx in range(len(class_names)):
#     class_name = class_names[class_idx]
#     class_df = validation_set[class_idx]
#     x_positive = class_df[class_df['belongs_to_class'] == 1].drop('belongs_to_class', axis=1)
#     x_negative = class_df[class_df['belongs_to_class'] == 0].drop('belongs_to_class', axis=1)
#
#     x_positive = np.array(x_positive)
#     x_negative = np.array(x_negative)
#
#     X_flatten_features = np.concatenate((x_positive, x_negative), axis=0)
#     y_positive = np.ones(len(x_positive), dtype=int)
#     y_negative = np.zeros(len(x_negative), dtype=int)
#     y_true = np.concatenate((y_positive, y_negative))
#
#     # {y_true} 0: doesn't belong - outlier , 1: belongs - normal
#     # {y_pred} -1: outlier, 1: normal
#     # task: change -1 to 0
#
#     y_pred = models[class_idx].predict(X_flatten_features)
#
#     # {y_true} 0: doesn't belong - outlier , 1: belongs - normal
#     # {y_pred} -1: outlier, 1: normal
#     # task: change -1 to 0
#     y_pred[y_pred == -1] = 0
#
#     # Compute the confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#
#     # Plot the confusion matrix using seaborn
#     plt.subplot(5, 3, class_idx + 1)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title(f'Confusion Matrix for Class: {class_name}')
#
# plt.tight_layout()
# plt.show()

def plot_histogram(anomaly_scores_positive, anomaly_scores_negative, class_name):
    plt.hist(anomaly_scores_positive, bins=50, label=f'Class {class_name} (Positive)', alpha=0.5)
    plt.hist(anomaly_scores_negative, bins=50, label=f'Class {class_name} (Negative)', alpha=0.5)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'Anomaly Score Distribution for Class: {class_name}')

plt.figure(figsize=(15, 10))
for class_idx in range(len(class_names)):
    class_name = class_names[class_idx]
    class_df = validation_set[class_idx]
    x_positive = class_df[class_df['belongs_to_class'] == 1].drop('belongs_to_class', axis=1)
    x_negative = class_df[class_df['belongs_to_class'] == 0].drop('belongs_to_class', axis=1)

    x_positive = np.array(x_positive)
    x_negative = np.array(x_negative)

    anomaly_scores_positive = models[class_idx].decision_function(x_positive)
    anomaly_scores_negative = models[class_idx].decision_function(x_negative)
    # anomaly_scores_positive = models[class_idx].score_samples(x_positive)
    # anomaly_scores_negative = models[class_idx].score_samples(x_negative)

    plt.subplot(5, 3, class_idx + 1)
    plot_histogram(anomaly_scores_positive, anomaly_scores_negative, class_name)

plt.tight_layout()
plt.show()

# Define a list of colors for each class
colors = ['blue', 'red', 'green', 'purple', 'orange',
          'cyan', 'magenta', 'yellow', 'brown', 'pink',
          'gray', 'lime', 'teal', 'indigo', 'olive']

plt.figure(figsize=(15, 10))
for class_idx in range(len(class_names)):
    class_name = class_names[class_idx]
    class_df = validation_set[class_idx]
    x_positive = class_df[class_df['belongs_to_class'] == 1].drop('belongs_to_class', axis=1)
    x_negative = class_df[class_df['belongs_to_class'] == 0].drop('belongs_to_class', axis=1)

    x_positive = np.array(x_positive)
    x_negative = np.array(x_negative)

    X_flatten_features = np.concatenate((x_positive, x_negative), axis=0)
    y_positive = np.ones(len(x_positive), dtype=int)
    y_negative = np.zeros(len(x_negative), dtype=int)
    y_true = np.concatenate((y_positive, y_negative))

    fpr, tpr, thresholds = roc_curve(y_true, models[class_idx].decision_function(X_flatten_features))
    # fpr, tpr, thresholds = roc_curve(y_true, models[class_idx].score_samples(X_flatten_features))
    roc_auc = auc(fpr, tpr)

    # Use a different color for each class's ROC curve
    plt.plot(fpr, tpr, color=colors[class_idx], lw=2, label=f'Class {class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Classes')
plt.legend(loc="lower right")
plt.show()
