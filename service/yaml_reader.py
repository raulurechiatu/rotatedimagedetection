import cv2
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import os
from scipy.ndimage import rotate
import util.constants as constants
import numpy as np
from objects.CardData import CardData
import service.image_processor as ip


# Method used to load the raw data from the yaml files
# this data will be parsed later
def load_yaml_data(folder_path, files_to_load=-1):
    yaml_data = []
    final_path = Path(__file__).parent / folder_path

    file_names = os.listdir(final_path)

    current_file = 0
    # The reason behind this if is to iterate more efficiently over the whole dataset without adding a check step
    # each time
    if files_to_load == -1:
        for file_name in file_names:
            with open(folder_path[3:] + file_name) as file:
                yaml_data.append(yaml.load(file, Loader=SafeLoader))

    else:
        for file_name in file_names:
            current_file += 1
            if current_file > files_to_load:
                break
            with open(folder_path[3:] + file_name) as file:
                yaml_data.append(yaml.load(file, Loader=SafeLoader))

    return yaml_data


# A method to parse the yaml file and return an object array
# this will split each image into arrays of rectangles
# if an image has 2 cards, with 4 corners showing the result of this method will be an array of 4 objects
# each containing cx, cy, width, height, angle and card_id
# after all the information was loaded a rectangle is obtained from the corners using the previous information
# this rectangle is resized to the parameters specified in the constants file in order to be usable by the cnn
# this will apply the previous logic on all the labels and images loaded
def parse_yaml_data(yaml_array, imgs):
    data = []
    labels = []
    total_cards = 0
    for yaml_entry in range(0, len(yaml_array)):
        yaml_data = yaml_array[yaml_entry]['BBmat']['data']
        current_data = []
        current_labels = []
        for current_property in range(0, len(yaml_data), 6):
            x = yaml_data[current_property]
            y = yaml_data[current_property+1]
            width = yaml_data[current_property+2]
            height = yaml_data[current_property+3]
            angle = yaml_data[current_property + 4]

            # this offset is needed because the rotation of the image yielded partially cropped images
            size_offset = 5
            half_height = int(height/2) + size_offset
            half_width = int(width/2) + size_offset

            corner_rect = imgs[yaml_entry][y - half_height:y + half_height, x - half_width:x + half_width]

            # If the corner failed to be obtained we just skip it
            if len(corner_rect) == 0:
                continue

            # rotation and resize of the image
            corner_rect = rotate(corner_rect, angle-90)
            corner_rect = ip.resize(corner_rect)

            card_id = yaml_data[current_property + 5]
            card_data = CardData(x, y,
                                 width, height,
                                 angle,
                                 card_id,
                                 corner_rect)
            current_data.append(card_data)

            if card_id not in current_labels:
                current_labels.append(card_id)

        imgs[yaml_entry] = ip.resize(imgs[yaml_entry])
        data.append(current_data)
        labels.append(current_labels)
        total_cards += len(current_data)
    return data, labels, total_cards


def parse_yaml_data_lite(yaml_array, imgs):
    labels = []
    for yaml_entry in range(0, len(yaml_array)):
        current_labels = []
        yaml_data = yaml_array[yaml_entry]['BBmat']['data']
        for current_property in range(0, len(yaml_data), 12):
            current_labels.append(yaml_data[current_property + 5])
        labels.append(current_labels)
        imgs[yaml_entry] = ip.resize(imgs[yaml_entry])
    return imgs, labels