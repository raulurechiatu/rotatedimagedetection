import service.image_loader as il
import service.yaml_reader as yr
import service.network_service as ns
from util.constants import images_to_load
import service.image_processor as ip


# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
yaml_data_path = "../resources/images/ssip_20k_cards/gt/"
assets_image_name = '2_of_clubs'

card_folder_path = "../resources/images/ssip_20k_cards/img/"


# will predict the solution for an image
# will reshape and resize the actual image to be suitable for prediction
def predict_image(path):
    model = ns.start_training(None, None)
    validation_image = ip.reshape(ip.resize(il.load_image_cv(path)))
    prediction_vector, (solution_id, solution_value) = ns.predict(model, validation_image)
    print("Predicted solution is:", solution_id, " with confidence level of ", solution_value)


def load_train():
    ns.gpu_setup()
    ns.gpu()
    card_images, card_names = il.load_images(card_folder_path, images_to_load)
    # image_data, labels, total_cards = yr.parse_yaml_data(yr.load_yaml_data(yaml_data_path, images_to_load), card_images)
    image_data, labels = yr.parse_yaml_data_lite(yr.load_yaml_data(yaml_data_path, images_to_load), card_images)

    # Start the training
    # model = ns.start_training(image_data, total_cards)
    model = ns.start_training(card_images, labels, len(card_images))


if __name__ == '__main__':
    # Segment
    # ip.segment_image(il.load_image_cv(card_folder_path + "CARD_00000.jpg", is_float32=False))
    # ip.segment_image(il.load_image_cv(card_folder_path + "CARD_00007.jpg", is_float32=False))
    ip.segment_image(il.load_image_cv(card_folder_path + "CARD_00016.jpg", is_float32=False))

    # Prediction
    # predict_image("../resources/images/validate/validate3.png")

    # il.display_image(image_data[0][0].rectangle)

    # Load and train data
    # load_train()
