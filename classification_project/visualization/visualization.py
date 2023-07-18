import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
        ax = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='Blues',
                         xticklabels=class_names, yticklabels=class_names)
