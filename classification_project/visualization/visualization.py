from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

class Visualization:
    @staticmethod
    def TSNE(features, labels, title=None):
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=0)

        test_representations_2d = tsne.fit_transform(features)

        # Create a plot
        plt.figure(figsize=(10,10))
        scatter = plt.scatter(test_representations_2d[:, 0], test_representations_2d[:, 1], c=labels.flatten(), cmap='tab10')

        # Create a legend with class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        legend1 = plt.legend(*scatter.legend_elements(num=10), title="Classes")
        plt.gca().add_artist(legend1)

        # Convert labels to class names
        class_indices = [int(label.get_text().split("{")[-1].split("}")[0]) for label in legend1.texts]
        for t, class_index in zip(legend1.texts, class_indices):
            t.set_text(class_names[class_index])

        plt.show()

    @staticmethod
    def Pareto(y, dict_keys):
        # Count the number of samples per class
        class_counts = np.bincount(y)

        # Calculate the percentage of each class
        class_percentages = (class_counts / len(y)) * 100

        # Sort the classes in descending order of their percentages
        sorted_indices = np.argsort(class_percentages)[::-1]
        sorted_classes = sorted_indices.tolist()
        sorted_percentages = class_percentages[sorted_indices]

        # Calculate the cumulative percentage
        cumulative_percentage = np.cumsum(sorted_percentages)

        # Plot the Pareto chart
        plt.figure(figsize=(8, 6))
        plt.bar([dict_keys[str(cls)] for cls in sorted_classes], sorted_percentages, color='blue')
        for i, cls in enumerate(sorted_classes):
            plt.text(i, sorted_percentages[i], str(class_counts[cls]), ha='center', va='bottom')

        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.title('Pareto Chart')
        plt.ylim(0, 100)  # Set the y-axis limits from 0 to 100
        plt.show()

    @staticmethod
    def Confusion_matrix(y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(len(class_names), len(class_names)))
        ax = sns.heatmap(cm, fmt=".0f", annot=True, cmap='Blues',
                         xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def show_downsampled_image(img, new_img):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[1].imshow(new_img)
        ax[1].set_title("New Image")
        plt.show()

    # Function to plot the ROC Curve
    @staticmethod
    def plot_roc_curve(model, X, y):
        # Assuming your model predicts probabilities, use predict_proba
        y_probs = model.predict(X)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_probs.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    # Function to plot the Precision-Recall Curve
    @staticmethod
    def plot_precision_recall_curve_multi_class(model, X, y):
           # Predict probabilities for each class
        y_prob = model.predict(X)

        # Compute precision-recall curve and average precision for each label (class)
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(y.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_prob[:, i])
            average_precision[i] = average_precision_score(y[:, i], y_prob[:, i])

        # Micro-average precision-recall curve and average precision
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y.ravel(), y_prob.ravel()
        )
        average_precision["micro"] = average_precision_score(y, y_prob, average="micro")

        # Plot precision-recall curves for each label (class)
        plt.figure()
        plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', lw=4,
                 label='Micro-average Precision-Recall curve (area = {0:0.2f})'.format(average_precision["micro"]))
        for i in range(y.shape[1]):
            plt.plot(recall[i], precision[i], lw=2,
                     label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Multi-Label Classification')
        plt.legend(loc='best')
        plt.show()

    @staticmethod
    def calculate_validation_metrics(model, X_val, y_val):
        y_prob = model.predict(X_val)
        y_pred = (y_prob > 0.5).astype(int)

        # Convert y_val to binary labels (0 or 1)
        y_val_binary = (y_val > 0.5).astype(int)

        accuracy = accuracy_score(y_val_binary, y_pred)
        loss = log_loss(y_val_binary, y_prob)
        f1 = f1_score(y_val_binary, y_pred, average='samples')  # Use average='samples' for multi-label
        return accuracy, loss, f1
    # Function to plot convergence graphs for a classification model
    @staticmethod
    def plot_convergence_graphs(history):
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(train_accuracy) + 1)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accuracy, 'o-', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'o-', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy vs. Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, 'o-', label='Training Loss')
        plt.plot(epochs, val_loss, 'o-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss vs. Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()