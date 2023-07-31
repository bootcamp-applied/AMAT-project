# import requests
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
#
# def download_and_resize_images(image_urls, resize_shape=(3, 32, 32)):
#     """
#     Download several images from provided URLs, plot them, and resize the images.
#
#     Parameters:
#         image_urls (list): A list of URLs of images to download.
#         resize_shape (tuple): The target shape to resize the images to. Default is (3, 32, 32).
#
#     Returns:
#         list: A list of NumPy arrays representing the resized images.
#     """
#
#     # Function to download an image from a given URL
#     def download_image(url):
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         return Image.open(response.raw)
#
#     # Download and plot the images
#     num_images = len(image_urls)
#     fig, axs = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
#
#     resized_images = []
#     for i, url in enumerate(image_urls):
#         # Download image
#         image = download_image(url)
#
#         # Convert image to NumPy array
#         image_array = np.array(image)
#
#         # Resize image
#         resized_image = image.resize(resize_shape[1:], Image.BICUBIC)
#         resized_images.append(np.array(resized_image))
#
#         # Plot resized image
#         if num_images > 1:
#             axs[i].imshow(np.array(resized_image))
#             axs[i].axis('off')
#             axs[i].set_title(f"Resized Image {i + 1}")
#         else:
#             axs.imshow(np.array(resized_image))
#             axs.axis('off')
#             axs.set_title(f"Resized Image {i + 1}")
#
#     plt.show()
#
#     return resized_images


import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def download_and_resize_images(image_urls, resize_shape=(3, 32, 32)):
    """
    Download several images from provided URLs, plot them, and resize the images.

    Parameters:
        image_urls (list): A list of URLs of images to download.
        resize_shape (tuple): The target shape to resize the images to. Default is (3, 32, 32).

    Returns:
        list: A list of NumPy arrays representing the resized images.
    """

    # Function to download an image from a given URL
    def download_image(url):
        response = requests.get(url, stream=True, verify=False)  # Disable SSL certificate verification
        response.raise_for_status()
        return Image.open(response.raw)

    # Download and plot the images
    num_images = len(image_urls)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 4, 4))

    resized_images = []
    for i, url in enumerate(image_urls):
        # Download image
        image = download_image(url)

        # Convert image to NumPy array
        image_array = np.array(image)

        # Resize image
        resized_image = image.resize(resize_shape[1:], Image.BICUBIC)
        resized_images.append(np.array(resized_image))

        # Plot resized image
        if num_images > 1:
            axs[i].imshow(np.array(resized_image))
            axs[i].axis('off')
            axs[i].set_title(f"Resized Image {i + 1}")
        else:
            axs.imshow(np.array(resized_image))
            axs.axis('off')
            axs.set_title(f"Resized Image {i + 1}")

    plt.show()

    return resized_images
