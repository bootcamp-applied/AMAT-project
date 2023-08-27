import matplotlib.pyplot as plt
import pandas as pd
import random

def plot_images(images):
    num_columns = 3

    # Calculate the number of rows required to display all images in the given number of columns
    num_rows = (len(images) + num_columns - 1) // num_columns

    # Create the grid of subplots
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 15))  # Adjust figsize as needed

    # Flatten the 2D array of subplots so that we can iterate through them easily
    axs = axs.ravel()

    # Iterate through images and display them in the subplots
    for idx, image in enumerate(images):
        axs[idx].imshow(image)
        axs[idx].axis('off')  # Turn off the axis to remove ticks and labels

    # Show the plot
    plt.show()


def load_new_images(num_rows=10):
    csv_file_path = '../../data/processed/cifar_100.csv'
    total_rows = sum(1 for line in open(csv_file_path)) - 1  # Subtract 1 for the header
    no_labels = [10, 11, 12, 13, 14]
    skip_rows = sorted(random.sample(range(1, total_rows + 1), total_rows - num_rows))

    df = pd.read_csv(csv_file_path, skiprows=skip_rows)
    filtered_df = df[~df['label'].isin(no_labels)]
    images = []
    for _, row in filtered_df.iterrows():
        pixel_columns = row.values[2:]
        image = pixel_columns.reshape(3, 32, 32).transpose(1, 2, 0)
        images.append(image)

    plot_images(images)
