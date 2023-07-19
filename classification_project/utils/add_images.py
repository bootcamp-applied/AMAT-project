from classification_project.utils.handling_new_image import dataNewImage

def useNewImage():
    add_image = dataNewImage()
    pathes_of_images = {0: r'../../data/raw/new_images/airplan.jpg',
                        1: r'../../data/raw/new_images/automobile.jpg',
                        2: r'../../data/raw/new_images/bird.jpg',
                        3: r'../../data/raw/new_images/cat.jpg',
                        4: r'../../data/raw/new_images/deer.jpg',
                        5: r'../../data/raw/new_images/dog.jpg',
                        6: r'../../data/raw/new_images/frog.jpg',
                        7: r'../../data/raw/new_images/horse.jpg',
                        8: r'../../data/raw/new_images/ship.jpg',
                        9: r'../../data/raw/new_images/truck.jpg',
                        10: r'../../data/raw/new_images/fish.jpg',
                        12: r'../../data/raw/new_images/flowers.jpg',
                        13: r'../../data/raw/new_images/trees.jpg',
                        14: r'../../data/raw/new_images/vegetables.jpg'}
    for label in pathes_of_images:
        add_image.read_add_to_df(pathes_of_images[label], label)