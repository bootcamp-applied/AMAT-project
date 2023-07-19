from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_curve, auc, precision_recall_curve

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
        ax = sns.heatmap(confusion_matrix(y_true, y_pred), fmt=".0f", annot=True, cmap='Blues',
                         xticklabels=class_names, yticklabels=class_names)

    @staticmethod
    def plot_learning_curve(history):
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
        plt.title('Learning Curve - Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, 'o-', label='Training Loss')
        plt.plot(epochs, val_loss, 'o-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve - Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Function to plot the ROC Curve
    @staticmethod
    def plot_roc_curve(model, X, y):
        y_prob = model.predict(X)
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

    # Function to plot the Precision-Recall Curve
    @staticmethod
    def plot_precision_recall_curve(model, X, y):
        y_prob = model.predict(X)
        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.show()

    # Function to calculate validation metrics
    @staticmethod
    def calculate_validation_metrics(model, X_val, y_val):
        y_prob = model.predict(X_val)
        y_pred = (y_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        loss = log_loss(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)
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