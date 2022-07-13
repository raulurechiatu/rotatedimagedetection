import os
import time
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

segmentation_path = "resources/images/output/segmentation/"


# A method used to load the images
# the user can change the actual library used to load images, currently the one used is: cv2
def load_images(folder_path, images_to_load=-1, offset=0):
    card_images = []
    final_path = Path(__file__).parent / folder_path

    before = time.time()
    image_names = os.listdir(final_path)
    after = time.time()
    print("Execution time for listdir(getting the images name) is ", (after-before), "s for ", len(image_names), " images")
    image_number = offset

    # The reason behind this if is to iterate more efficiently over the whole dataset without adding a check step
    # each time
    if images_to_load == -1:
        for image_name in image_names:
            # Here we can change the library used to load images in memory
            card_images.append(load_image_cv(folder_path + image_name))

    else:
        for image_name in image_names:
            image_number = image_number + 1
            # Here we can change the library used to load images in memory
            card_images.append(load_image_cv(folder_path + image_name))
            if images_to_load == image_number:
                break

    after_image_load = time.time()
    print("Execution time for loading images in memory is ", (after_image_load-after), "seconds for ", images_to_load, " images")
    # print(card_images)

    return card_images, image_names


# 10k images: 24.8s
# 1k images: 2.3s
# Used for an efficient way to load the images
def load_image_matplot(path):
    # Load the image
    final_path = Path(__file__).parent / path
    original = mpimg.imread(final_path)
    return original


# 1k images: 2.3s
def load_image_pil(path):
    final_path = Path(__file__).parent / path
    original = Image.open(final_path)
    return list(original.getdata())


# ~1000 x ~750
def load_image_cv(path, is_float32=True):
    path = path[3:]
    # return cv2.imread(path, 0)
    if is_float32:
        return cv2.imread(path, 0).astype('float32') / 255
    else:
        return cv2.imread(path)


# displays an image using matplotlib
def display_image(image, title=""):
    f, axarr = plt.subplots(1, 1)
    # Display the image
    axarr.imshow(image, cmap='gray')
    if title != "":
        plt.title(title)
    plt.show()