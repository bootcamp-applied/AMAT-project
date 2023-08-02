# import pandas as pd
# import albumentations as A
# import numpy as np
# import os
#
# class Argumentation:
#     def load_data():
#         file_path = '../../data/processed/cifar-100.csv'
#
#         # Replace 'label col' with the actual name of the column containing the labels in your CSV file
#         label_column = 'label'
#
#         # List of label values to read
#         labels_to_read = [1, 2, 17, 13, 4]
#
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv(file_path)
#
#         # Filter the rows based on the values in the label column
#         filtered_df = df[df[label_column].isin(labels_to_read)]
#         return filtered_df
#
#     def apply_augmentation(images):
#         # Define augmentation transformations
#         transform = A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.Rotate(limit=15),
#             A.RandomBrightnessContrast(p=0.2),
#         ])
#         augmented_images = [transform(image=image)['image'] for image in images]
#         return np.array(augmented_images)
#
#     def create_new_df(new_images,filtered_df):
#         num_images = new_images.shape[0]
#
#         is_train = filtered_df.iloc[:, 0:1]
#
#         # Assuming 'labels' is a separate array or list containing the labels for each image
#         labels = filtered_df.iloc[:, 1:2]
#
#         # Assuming 'labels' is a NumPy array or list containing the labels for each image
#         labels = np.array(labels)  # Replace this with your actual 'labels' array or list
#         is_train = np.array(is_train)
#
#         # Create a DataFrame from the new_images array
#         df = pd.DataFrame(new_images.reshape(num_images, -1))
#
#         # Normalize the pixel values to be between 0 and 255 and convert to integers
#         df = (df * 255).astype(int)
#
#         # Add the 'is_train' and 'label' columns to the beginning of the DataFrame
#         df.insert(0, 'is_train', is_train)
#         df.insert(1, 'label', labels)
#         return df
#
#
#     filtered_df=load_data()
#     is_train=filtered_df.iloc[:,0:1]
#     labels=filtered_df.iloc[:,1:2]
#     pixels=filtered_df.iloc[:,2:]
#
#     # Assuming pixels is a NumPy array with shape (15000, 3072)
#     pixels = np.array(pixels)
#
#     # Split the 'pixels' array into a list of 15000 images
#     num_images = pixels.shape[0]
#     images_list = [pixels[i].reshape(3, 32, 32) for i in range(num_images)]
#
#     # Normalize pixel values to the range [0, 1]
#     images_list = [image.astype(np.float32) / 255.0 for image in images_list]
#
#     # Apply augmentation to all 15000 images
#     new_images = apply_augmentation(images_list)
#
#     #convert to df
#     df=create_new_df(new_images,filtered_df)
#
#     #write the augmentation df to csv
#     path = 'data/processed/cifar-100_augmentations.csv'
#     if os.path.exists(path):
#         # If the file exists, delete it
#         os.remove(path)
#
#     df.to_csv(path, index=False, encoding='utf-8', mode='w')
#
#
import pandas as pd
import albumentations as A
import numpy as np
import os

class Augmentation:
    @staticmethod
    def load_data(file_path, label_column, labels_to_read):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Filter the rows based on the values in the label column
        filtered_df = df[df[label_column].isin(labels_to_read)]
        return filtered_df

    @staticmethod
    def apply_augmentation(images):
        # Define augmentation transformations
        # transform = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     A.Rotate(limit=15),
        #     A.RandomBrightnessContrast(p=0.2),
        # ])
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.1),  # Apply rotation to 10% of the images
            A.RandomBrightnessContrast(p=0.2),
        ])
        augmented_images = [transform(image=image)['image'] for image in images]
        return np.array(augmented_images)

    @staticmethod
    def create_new_df(new_images, filtered_df):
        num_images = new_images.shape[0]

        is_train = filtered_df.iloc[:, 0:1]

        # Assuming 'labels' is a separate array or list containing the labels for each image
        labels = filtered_df.iloc[:, 1:2]

        # Assuming 'labels' is a NumPy array or list containing the labels for each image
        labels = np.array(labels)  # Replace this with your actual 'labels' array or list
        is_train = np.array(is_train)

        # Create a DataFrame from the new_images array
        df = pd.DataFrame(new_images.reshape(num_images, -1))

        # Normalize the pixel values to be between 0 and 255 and convert to integers
        df = (df * 255).astype(int)

        # Add the 'is_train' and 'label' columns to the beginning of the DataFrame
        df.insert(0, 'is_train', is_train)
        df.insert(1, 'label', labels)
        return df

    @staticmethod
    def main():
        file_path = '../../data/processed/cifar-100.csv'
        label_column = 'label'
        labels_to_read = [1, 2, 17, 14, 4]

        # Load the data
        filtered_df = Augmentation.load_data(file_path, label_column, labels_to_read)

        is_train = filtered_df.iloc[:, 0:1]
        labels = filtered_df.iloc[:, 1:2]
        pixels = filtered_df.iloc[:, 2:]

        # Assuming pixels is a NumPy array with shape (15000, 3072)
        pixels = np.array(pixels)

        # Split the 'pixels' array into a list of 15000 images
        num_images = pixels.shape[0]
        images_list = [pixels[i].reshape(3, 32, 32) for i in range(num_images)]

        # Normalize pixel values to the range [0, 1]
        images_list = [image.astype(np.float32) / 255.0 for image in images_list]

        # Apply augmentation to all 15000 images
        new_images = Augmentation.apply_augmentation(images_list)

        # Convert to df
        df = Augmentation.create_new_df(new_images, filtered_df)

        # Write the augmentation df to csv
        path = '../../data/processed/augmentation.csv'
        if os.path.exists(path):
            # If the file exists, delete it
            os.remove(path)

        df.to_csv(path, index=False, encoding='utf-8', mode='w')

if __name__ == "__main__":
    Augmentation.main()
