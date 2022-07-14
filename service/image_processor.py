import cv2
import numpy as np
import service.image_loader as il
import service.network_service as ns
import util.constants as constants
from scipy.ndimage import rotate
import util.app_helper as ah

assets_parent_path = "../resources/images/ssip_20k_cards/assets/"
card_folder_path = "../resources/images/ssip_20k_cards/img/"

find_corners_from_side = 30
corner_crop_size = 40


def segment_image(img, model):
    img = cv2.threshold(img, 160, 255, cv2.THRESH_TOZERO)[1]
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # draw rectangles
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)

        crop, crop_point = get_cropped_card(img, cnt)
        corner_points = get_corners(crop)
        corners = get_corner_cropped(img, crop, crop_point, corner_points, model)

    return None, img


def get_cropped_card(img, cnt):
    space_from_margin = 5
    mask = np.zeros_like(img)
    cv2.drawContours(mask, cnt, -1, (255, 0, 0), 3)
    out = np.zeros_like(img)  # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img[topy - space_from_margin:bottomy + space_from_margin,
          topx - space_from_margin:bottomx + space_from_margin]
    return out, (topx, topy)


def get_corners(crop):
    if len(crop) == 0 or len(crop[0]) == 0:
        return []

    canny = cv2.Canny(crop, 150, 255)

    corners = cv2.goodFeaturesToTrack(canny, 6, 0.01, 50)
    centers = []
    if len(corners) < 1:
        return centers

    for corner in corners:
        x, y = corner.ravel()
        if find_corners_from_side < x < crop.shape[0] - find_corners_from_side and \
                find_corners_from_side < y < crop.shape[1] - find_corners_from_side:
            continue
        centers.append((y, x))
        cv2.circle(canny, (int(x), int(y)), 10, (255, 0, 0), -1)
    # cv2.imshow("Corner Points", crop)
    cv2.imshow("Corner Points", canny)
    cv2.waitKey(40)
    return centers


def get_corner_cropped(img, crop, crop_point, corner_points, model):

    corners = []
    max_angle_prediction = 0
    max_angle_class = 0
    max_angle = -1
    for pt in corner_points:
        x1 = 0 if pt[0] - corner_crop_size < 0 else int(pt[0] - corner_crop_size)
        y1 = crop.shape[0] if pt[0] + corner_crop_size > crop.shape[0] else int(pt[0] + corner_crop_size)

        x2 = 0 if pt[1] - corner_crop_size < 0 else int(pt[1] - corner_crop_size)
        y2 = crop.shape[1] if pt[1] + corner_crop_size > crop.shape[1] else int(pt[1] + corner_crop_size)

        corner = crop[x1:y1, x2:y2]
        # il.display_image(corner)
        # cv2.imshow("Corners Img", corner)
        corners.append(corner)
        # cv2.imshow("Fed Image", corner)
        # cv2.waitKey(40)

        # for angle in [0]:[0, 45, 90, 135, 180, 225, 270]
        for angle in range(0, 360, 10):
            validation_image = reshape(resize(rotate(corner, angle)))/255
            cv2.imshow("Rotated Image", validation_image[0])
            cv2.waitKey(10)
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

    return corners


def segment_image_2(img, model):
    img = cv2.threshold(img, 170, 255, cv2.THRESH_TOZERO)[1]
    # il.display_image(img)

    rectangle_size = 20
    same_rectangle_offset = 40
    kernel_sizes = [10, 30]
    sizes = [20, 30]
    # angles = [0, 90, 180, 270]
    angles = [0]

    locs = get_template_locs(img, sizes, angles)

    current_pt0 = []
    current_pt1 = []
    results = []
    solution_ids = []

    for loc_id in range(0, len(locs)):
        for pt in zip(*locs[loc_id][::-1]):
            if check_if_point_exists(current_pt0, current_pt1, pt, same_rectangle_offset):
                continue

            current_pt0.append(pt[0])
            current_pt1.append(pt[1])
            # plt.imshow(cv2.cvtColor(img[pt[1] - h:pt[1] + h, pt[0] - h:pt[0] + h], cv2.COLOR_BGR2RGB))
            # plt.show()

            max_angle_prediction = 0
            max_angle_class = 0
            max_angle_pt = []

            for kernel_size in kernel_sizes:
                result = img[pt[1] - kernel_size:pt[1] + kernel_size, pt[0] - kernel_size:pt[0] + kernel_size] / 255
                results.append(result)
                il.display_image(result)
                for angle in angles:
                    if len(result) == 0 or len(result[0]) == 0:
                        continue

                    validation_image = reshape(resize(rotate(result, angle)))
                    prediction_vector, (solution_id, solution_value) = ns.predict(model, validation_image)
                    if solution_value > max_angle_prediction:
                        max_angle_prediction = solution_value
                        max_angle_class = solution_id
                        max_angle_pt = pt

            if max_angle_prediction < 0.7:
                continue

            rect = cv2.rectangle(img,
                                 (max_angle_pt[0] - rectangle_size, max_angle_pt[1] - rectangle_size),
                                 (max_angle_pt[0] + rectangle_size, max_angle_pt[1] + rectangle_size),
                                 (255, 0, 0),
                                 2)
            cv2.putText(img,
                        str("%s (%d-%f)" % (ah.get_card_from_id(max_angle_class), max_angle_class, round(max_angle_prediction, 2))),
                        (pt[0] - kernel_size, pt[1] - rectangle_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # if solution_id not in solution_ids:
            solution_ids.append((ah.get_card_from_id(max_angle_class), max_angle_prediction))
            print("Predicted solution is:", ah.get_card_from_id(max_angle_class), " with confidence level of ", max_angle_prediction)

            # cropped = rotate(cropped, loc_id % len(angles) * -90)

    print("Cards identified in the picture are: ", solution_ids)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    return results, img


def get_template_locs(img, sizes, angles):
    threshold = 0.5
    locs = []  # create an empty tuple

    templates = ["4_of_clubs.png", "4_of_spades.png", "4_of_hearts.png", "4_of_diamonds.png"]
    for template_card in templates:
        template = il.load_image_cv(assets_parent_path + template_card, is_float32=False)
        template = cv2.threshold(template, 150, 255, cv2.THRESH_TOZERO)[1]
        tmp = template[90:175, 10:95]
        # il.display_image(tmp)

        for size in sizes:
            tmp = resize(tmp, width=size, height=size)

            for angle in angles:
                if angle == 0:
                    res = cv2.matchTemplate(img, tmp, cv2.TM_CCOEFF_NORMED)
                else:
                    res = cv2.matchTemplate(img, rotate(tmp, angle), cv2.TM_CCOEFF_NORMED)
                pt = np.where(res >= threshold)

                if pt[0].size != 0:
                    locs.append(pt)

    return locs


# Checks if the identified rectangle is similar as the one before
def check_if_point_exists(current_pt0, current_pt1, pt, same_rectangle_offset):
    for current_pt_id in range(0, len(current_pt0)):
        if current_pt0[current_pt_id] - same_rectangle_offset < pt[0] < current_pt0[
            current_pt_id] + same_rectangle_offset and \
                current_pt1[current_pt_id] - same_rectangle_offset < pt[1] < current_pt1[
            current_pt_id] + same_rectangle_offset \
                :
            # and current_threshold < thresholds[current_pt_id]\
            return True
    return False


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
