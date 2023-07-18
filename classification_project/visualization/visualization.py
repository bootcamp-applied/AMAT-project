import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
# import seaborn as sns


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
