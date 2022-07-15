import cv2
import numpy as np
import service.network_service as ns
import util.constants as constants
from scipy.ndimage import rotate
import util.app_helper as ah
import util.game as game

assets_parent_path = "../resources/images/ssip_20k_cards/assets/"
card_folder_path = "../resources/images/ssip_20k_cards/img/"

find_corners_from_side = 30
corner_crop_size = 50


# The starting point of the algorithm to classify a real image
# in this we will find multiple steps applied to the original image like thresholding, finding contours,
# extracting corners, rotating, resizing, reshaping and prediction, all working together on the same image
def segment_image(img, model):
    if constants.play_game:
        game.reset_score()

    img = cv2.threshold(img, 170, 255, cv2.THRESH_TOZERO)[1]

    # Obtaining the contours from the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cards_in_image = []
    # draw rectangles
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # if the contour area is lesser than 4000, it will probably be something other than a card or set of cards
        if area < 4000:
            continue
        cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)

        crop, crop_point = get_cropped_card(img, cnt)
        corner_points = get_corners(crop)
        corners, card_class = get_corner_cropped(img, crop, crop_point, corner_points, model)
        if constants.play_game:
            cards_in_image.append(card_class)
            game.compute_score(card_class)

    return img, cards_in_image


# Gets the cropped card out of the contour
# this will return the matrix representing only one contour cropped out of the original image and the (x1, y1) point
# that represents the coordinates of the left upper corner, later used for text rendering
def get_cropped_card(img, cnt):
    space_from_margin = 5
    mask = np.zeros_like(img)
    cv2.drawContours(mask, cnt, -1, (255, 0, 0), 3)
    out = np.zeros_like(img)
    # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img[topy - space_from_margin:bottomy + space_from_margin,
          topx - space_from_margin:bottomx + space_from_margin]
    return out, (topx, topy)


# Method used to obtain corners from an image, this will return the center point of all the corners detected
# this uses an improved version of harrison corner detector
def get_corners(crop):
    if len(crop) == 0 or len(crop[0]) == 0:
        return []

    # we firstly apply the canny edge detection algorithm on the image for better corner recognition
    canny = cv2.Canny(crop, 120, 255)

    # obtain the corners
    corners = cv2.goodFeaturesToTrack(canny, 6, 0.1, 50)
    centers = []
    if len(corners) < 1:
        return centers

    for corner in corners:
        x, y = corner.ravel()
        # if the corners are near the edge they are valid corners due to the cropping done previously
        # however the algorithm find corners in the middle of the image sometimes, so we added this check to see if
        # the points identified as corners are near the edge of the image or not
        if find_corners_from_side < x < crop.shape[0] - find_corners_from_side and \
                find_corners_from_side < y < crop.shape[1] - find_corners_from_side:
            continue
        centers.append((y, x))
        cv2.circle(canny, (int(x), int(y)), 10, (255, 0, 0), -1)
    # cv2.imshow("Corner Points", crop)
    cv2.imshow("Corner Points", canny)
    cv2.waitKey(20)
    return centers


# initially created to obtain the rectangles containing the number and the symbol, extended the usability in order
# to make the algorithm more efficient from a computing perspective
# in the method after the corner matrices are obtained we fed them to the CNN for it to predict the outcome
# in order to solve the rotation problem, we will rotate the obtained crop of the corner by 30 degrees between 0 and 360
# we compute a solution for each scenario and keep the best one out of them all
def get_corner_cropped(img, crop, crop_point, corner_points, model):

    corners = []
    max_angle_prediction = 0
    max_angle_class = -1
    max_angle = -1
    for pt in corner_points:
        x1 = 0 if pt[0] - corner_crop_size < 0 else int(pt[0] - corner_crop_size)
        y1 = crop.shape[0] if pt[0] + corner_crop_size > crop.shape[0] else int(pt[0] + corner_crop_size)

        x2 = 0 if pt[1] - corner_crop_size < 0 else int(pt[1] - corner_crop_size)
        y2 = crop.shape[1] if pt[1] + corner_crop_size > crop.shape[1] else int(pt[1] + corner_crop_size)

        corner = crop[x1:y1, x2:y2]
        corners.append(corner)
        # cv2.imshow("Fed Image", corner)
        # cv2.waitKey(40)

        # for angle in [0]:[0, 45, 90, 135, 180, 225, 270]
        for angle in range(0, 360, 30):
            validation_image = reshape(resize(rotate(corner, angle)))/255
            # cv2.imshow("Rotated Image", validation_image[0])
            # cv2.waitKey(5)
            prediction_vector, (solution_id, solution_value) = ns.predict(model, validation_image)
            if solution_value > max_angle_prediction:
                max_angle_prediction = solution_value
                max_angle_class = solution_id
                max_angle = angle

    if max_angle_prediction > 0:
        print("Predicted solution is:", ah.get_card_from_id(max_angle_class),
              "(", max_angle_class, max_angle_prediction, ") - ", max_angle)

        cv2.putText(img,
                    str("%s (%.2f)" % (ah.get_card_from_id(max_angle_class), max_angle_prediction)),
                    (int(crop_point[0]), int(crop_point[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    else:
        print("Solution not found, closest card was:", ah.get_card_from_id(max_angle_class),
              "(", max_angle_prediction, ") - ", max_angle)

    return corners, max_angle_class


# method used to grayscale an image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# A resize method used to generalize the size of an image
# needed because the neural network was trained on a specific size
def resize(img, width=constants.training_image_width, height=constants.training_image_height):
    return cv2.resize(img, (width, height),
                      interpolation=cv2.INTER_AREA)


# A reshape method used to generalize the shape needed for the network
def reshape(img):
    data = np.empty((1, constants.training_image_height, constants.training_image_width),
                    dtype='float32')
    data[0] = img
    return np.reshape(data, (1, constants.training_image_height, constants.training_image_width, 1))
