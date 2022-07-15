import cv2

import service.image_loader as il
import service.yaml_reader as yr
import service.network_service as ns
from util.constants import images_to_load
import service.image_processor as ip
import util.constants as constants
import util.app_helper as ah
import util.game as game


# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
yaml_data_path = "../resources/images/ssip_20k_cards/gt/"
assets_image_name = '2_of_clubs'
card_folder_path = "../resources/images/ssip_20k_cards/img/"


# will predict the solution for an image
# will reshape and resize the actual image to be suitable for prediction
def predict_image(path):
    model = ns.start_training(None, None, None)
    img = ip.resize(il.load_image_cv(path))
    thresh = cv2.threshold(img, .6, 1, cv2.THRESH_TOZERO)[1]
    validation_image = ip.reshape(thresh)
    prediction_vector, (solution_id, solution_value) = ns.predict(model, validation_image)
    print("Predicted solution is:", ah.get_card_from_id(solution_id), " with confidence level of ", solution_value)


def load_train():
    ns.gpu_setup()
    ns.gpu()
    image_data, total_cards = il.load_images_and_yaml(card_folder_path, yaml_data_path, images_to_load, constants.offset)

    # Start the training
    model = ns.start_training(image_data, None, total_cards)


def video_process():

    # cap = cv2.VideoCapture('../resources/images/ssip_20k_cards/sequence_for_test.mp4')
    cap = cv2.VideoCapture(0)
    model = ns.start_training(None, None, None)
    while cap.isOpened():
        ret, frame = cap.read()
        frame_original = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img, cards_in_image = ip.segment_image(gray, model)
        if constants.play_game:
            game.play_card(frame, cards_in_image)
            game.show_score(frame)
        cv2.imshow("Gray", gray)
        cv2.imshow("Thresholded", img)
        cv2.imshow("Game Board", frame)
        cv2.imshow("Live Camera", frame_original)

        key = cv2.waitKey(100)
        if key % 256 == 32:
            game.reset_generated_cards()

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Segment
    # ip.segment_image(il.load_image_cv(card_folder_path + "CARD_00000.jpg", is_float32=False))
    # ip.segment_image(il.load_image_cv(card_folder_path + "CARD_00007.jpg", is_float32=False))
    # ip.segment_image(il.load_image_cv(card_folder_path + "CARD_00016.jpg", is_float32=False))
    # ip.segment_image(il.load_image_cv("../resources/images/validate/validate3.png", is_float32=False))

    # Process the video
    video_process()

    # Prediction
    # predict_image("../resources/images/validate/validate1.png")
    # predict_image("../resources/images/validate/validate4.png")

    # il.display_image(image_data[0][0].rectangle)

    # Load and train data
    # load_train()
