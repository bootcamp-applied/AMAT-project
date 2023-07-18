import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class Visualization:
    @staticmethod
    def TSNE(X, y, title=None):
        # Perform TSNE on the data
        tsne = TSNE(n_components=2, random_state=42, perplexity=45)
        X_embedded = tsne.fit_transform(X)
        # Create a scatter plot of the data points with different colors for each class
        plt.figure(figsize=(8, 6))
        classes = np.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        for i, c in zip(classes, colors):
            plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], color=c, label=str(i))
        plt.legend(title='Class')
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
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
        # plt.bar(sorted_classes, sorted_percentages, color='blue')
        #         plt.plot(sorted_classes, cumulative_percentage, 'r-o')
        plt.bar([dict_keys[str(cls)] for cls in sorted_classes], sorted_percentages, color='blue')
        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.title('Pareto Chart')
        plt.ylim(0, 100)  # Set the y-axis limits from 0 to 100
        plt.show()

    @staticmethod
    def Confusion_matrix(y_true, y_pred, classes):
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()
