import cv2
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import os
from scipy.ndimage import rotate
import util.constants as constants


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


def parse_yaml_data(yaml_array, imgs):
    data = []
    total_cards = 0
    for yaml_entry in range(0, len(yaml_array)):
        yaml_data = yaml_array[yaml_entry]['BBmat']['data']
        current_data = []
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

            if len(corner_rect) == 0:
                continue

            corner_rect = rotate(corner_rect, angle-90)
            corner_rect = cv2.resize(corner_rect, (constants.training_image_width, constants.training_image_height),
                                     interpolation=cv2.INTER_AREA)

            card_data = CardData(x, y,
                                 width, height,
                                 angle,  # angle
                                 yaml_data[current_property + 5],  # card_id
                                 corner_rect)
            current_data.append(card_data)
        data.append(current_data)
        total_cards += len(current_data)
    return data, total_cards


class CardData:
    def __init__(self, cx, cy, width, height, angle, card_id, rectangle):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.angle = angle
        self.card_id = card_id
        self.rectangle = rectangle

    def __str__(self):
        print("cx: ", self.cx, " cy: ", self.cy, " width: ", self.width, " height: ", self.height,
              " angle: ", self.angle, " card_id: ", self.card_id)
