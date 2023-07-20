import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from classification_project.preprocessing.preprocessing import Preprocessing
from classification_project.models.cnn import CNN

df = pd.read_csv('../../data/processed/cifar-10-100.csv')
preprocessing = Preprocessing(df)
preprocessing.prepare_data()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(one_hot_encoder=True)

loaded_model = CNN.load_cnn_model('../saved_model/saved_cnn_model.keras')
loaded_history_model = CNN.load_cnn_history('../saved_model/saved_cnn_history.pkl')

def plot_correct_and_incorrect_classified_images(x_test, y_test, model, num_of_examples=9):
    # Get the model predictions for the entire test set
    y_pred = model.predict(x_test)
    predictions = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Get indices of correctly and incorrectly classified samples
    correctly_classified_indices = np.where(predictions == true_labels)[0]
    incorrectly_classified_indices = np.where(predictions != true_labels)[0]

    correctly_classified_indices = correctly_classified_indices[:num_of_examples]
    incorrectly_classified_indices = incorrectly_classified_indices[:num_of_examples]

    # Map class number to class label
    dict_path = '../utils/dict.json'
    with open(dict_path, 'r') as f:
      label_dict = json.load(f)

    # Plot correctly classified images
    num_columns_correct = 3
    num_rows_correct = (len(correctly_classified_indices) + num_columns_correct - 1) // num_columns_correct
    fig_correct, axs_correct = plt.subplots(num_rows_correct, num_columns_correct, figsize=(num_columns_correct*3, 3*num_rows_correct))
    axs_correct = axs_correct.ravel()

    for idx, idx_image in enumerate(correctly_classified_indices[:num_columns_correct * num_rows_correct]):
        image = x_test[idx_image]
        true_class_num = true_labels[idx_image]
        true_class_label = label_dict[str(true_class_num)]
        axs_correct[idx].imshow(image)
        axs_correct[idx].set_title(f'True label and predicted: {true_class_label}', fontsize=10)
        axs_correct[idx].axis('off')

    plt.suptitle('Correctly Classified Images')
    plt.show()

    # Plot incorrectly classified images
    num_columns_incorrect = 3
    num_rows_incorrect = (len(incorrectly_classified_indices) + num_columns_incorrect - 1) // num_columns_incorrect
    fig_incorrect, axs_incorrect = plt.subplots(num_rows_incorrect, num_columns_incorrect, figsize=(num_columns_incorrect*3, 3*num_rows_incorrect))
    axs_incorrect = axs_incorrect.ravel()

    for idx, idx_image in enumerate(incorrectly_classified_indices[:num_columns_incorrect * num_rows_incorrect]):
        image = x_test[idx_image]
        true_class_num = true_labels[idx_image]
        predicted_class_num = predictions[idx_image]
        prob = y_pred[idx_image][predicted_class_num]
        # prob = 0.9999999999
        probability = np.round(prob,3)

        true_class_label = label_dict[str(true_class_num)]
        predicted_class_label = label_dict[str(predicted_class_num)]
        axs_incorrect[idx].imshow(image)
        axs_incorrect[idx].set_title(f'True label: {true_class_label}\nPredicted label: {predicted_class_label}\nWith probability of: {probability:.3f} ', fontsize=10)
        axs_incorrect[idx].axis('off')

    plt.suptitle('Incorrectly Classified Images')
    plt.tight_layout()
    plt.show()
# def plot_correct_and_incorrect_classified_images(x_test, y_test, model, num_of_examples=9):
#     # Get the model predictions for the entire test set
#     predictions = np.argmax(model.predict(x_test), axis=1)
#     true_labels = np.argmax(y_test, axis=1)
#
#     # Get indices of correctly and incorrectly classified samples
#     correctly_classified_indices = np.where(predictions == true_labels)[0]
#     incorrectly_classified_indices = np.where(predictions != true_labels)[0]
#
#     correctly_classified_indices = correctly_classified_indices[:num_of_examples]
#     incorrectly_classified_indices = incorrectly_classified_indices[:num_of_examples]
#
#     # Map class number to class label
#     dict_path = '../utils/dict.json'
#     with open(dict_path, 'r') as f:
#       label_dict = json.load(f)
#
#     # Plot correctly classified images
#     num_columns_correct = 3
#     num_rows_correct = (len(correctly_classified_indices) + num_columns_correct - 1) // num_columns_correct
#     fig_correct, axs_correct = plt.subplots(num_rows_correct, num_columns_correct, figsize=(num_columns_correct*3, 3*num_rows_correct))
#     axs_correct = axs_correct.ravel()
#
#     for idx, idx_image in enumerate(correctly_classified_indices[:num_columns_correct * num_rows_correct]):
#         image = x_test[idx_image]
#         true_class_num = true_labels[idx_image]
#         true_class_label = label_dict[str(true_class_num)]
#         axs_correct[idx].imshow(image)
#         axs_correct[idx].set_title(f'True label and predicted: {true_class_label}', fontsize=10)
#         axs_correct[idx].axis('off')
#
#     plt.suptitle('Correctly Classified Images')
#     plt.show()
#
#     # Plot incorrectly classified images
#     num_columns_incorrect = 3
#     num_rows_incorrect = (len(incorrectly_classified_indices) + num_columns_incorrect - 1) // num_columns_incorrect
#     fig_incorrect, axs_incorrect = plt.subplots(num_rows_incorrect, num_columns_incorrect, figsize=(num_columns_incorrect*3, 3*num_rows_incorrect))
#     axs_incorrect = axs_incorrect.ravel()
#
#     for idx, idx_image in enumerate(incorrectly_classified_indices[:num_columns_incorrect * num_rows_incorrect]):
#         image = x_test[idx_image]
#         true_class_num = true_labels[idx_image]
#         predicted_class_num = predictions[idx_image]
#         true_class_label = label_dict[str(true_class_num)]
#         predicted_class_label = label_dict[str(predicted_class_num)]
#         axs_incorrect[idx].imshow(image)
#         axs_incorrect[idx].set_title(f'True label: {true_class_label}\nPredicted label: {predicted_class_label}', fontsize=10)
#         axs_incorrect[idx].axis('off')
#
#     plt.suptitle('Incorrectly Classified Images')
#     plt.show()

# Example usage:
plot_correct_and_incorrect_classified_images(x_test, y_test, loaded_model)

