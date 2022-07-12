import service.image_loader as il
import service.yaml_reader as yr
import service.network_service as ns
from util.constants import images_to_load
import gc

# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
assets_parent_path = "../resources/images/ssip_20k_cards/assets/"
yaml_data_path = "../resources/images/ssip_20k_cards/gt/"
assets_image_name = '2_of_clubs'

card_folder_path = "../resources/images/ssip_20k_cards/img/"

if __name__ == '__main__':
    ns.gpu_setup()
    card_images, card_names = il.load_images(card_folder_path, images_to_load)
    image_data, total_cards = yr.parse_yaml_data(yr.load_yaml_data(yaml_data_path, images_to_load), card_images)

    # for i in range(0, images_to_load):
        # ip.contour_image(card_images[i])
        # il.display_image(card_images[i], card_names[i])
        # print(image_data[i])

    # rectangles = ip.get_corner_rectangles(card_images, yaml_data)
    # print(len(rectangles))
    gc.collect()
    ns.start_training(image_data, total_cards)
