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


# will predict the solution for an image
# will reshape and resize the actual image to be suitable for prediction
def predict_image(path):
    model = ns.start_training(None, None)
    validation_image = yr.reshape(yr.resize(il.load_image_cv(path)))
    prediction_vector, (solution_id, solution_value) = ns.predict(model, validation_image)
    print("Predicted solution is:", solution_id, " with confidence level of ", solution_value)


if __name__ == '__main__':
    # predict_image("../resources/images/validate/validate1.png")

    ns.gpu_setup()
    # ns.cpu()
    ns.gpu()
    card_images, card_names = il.load_images(card_folder_path, images_to_load)
    image_data, total_cards = yr.parse_yaml_data(yr.load_yaml_data(yaml_data_path, images_to_load), card_images)

    # ip.contour_image(card_images[i])

    il.display_image(image_data[0][0].rectangle)
    model = ns.start_training(image_data, total_cards)
