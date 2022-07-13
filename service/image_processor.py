import cv2
import numpy as np
import pytesseract
import service.image_loader as il
import matplotlib.pyplot as plt
import util.constants as constants
from scipy.ndimage import rotate

assets_parent_path = "../resources/images/ssip_20k_cards/assets/"
card_folder_path = "../resources/images/ssip_20k_cards/img/"


def segment_image(img):
    # cv2.imshow('image', img)
    # cv2.waitKey()
    # contour_image(img)
    # asset_images = get_assets()
    h, w = 80, 50
    w_offset = 10
    same_rectangle_offset = 50
    sizes = [20, 25, 30, 40]
    angles = [0, 90, 180, 270]

    locs = get_template_locs(img, sizes, angles)

    current_pt0 = []
    current_pt1 = []
    for loc_id in range(0, len(locs)):
        for pt in zip(*locs[loc_id][::-1]):
            # rect = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            # tmp_img = rotate(img, loc_id*90)

            if check_if_point_exists(current_pt0, current_pt1, pt, same_rectangle_offset):
                continue

            current_pt0.append(pt[0])
            current_pt1.append(pt[1])
            # rect = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

            # if the rectangle was found horizontally or vertically, adjust the cropping size
            if loc_id % len(angles) == 0:
                cropped = img[pt[1] - int(h / 2):pt[1] + int(h / 2), pt[0] - w_offset:pt[0] + w - w_offset]
            elif loc_id % len(angles) == 1:
                cropped = img[pt[1] - w_offset:pt[1] + w - w_offset, pt[0] - int(h / 2):pt[0] + int(h / 2)]
            elif loc_id % len(angles) == 2:
                cropped = img[pt[1]:pt[1] + h, pt[0] - w_offset:pt[0] + w - w_offset]
            elif loc_id % len(angles) == 3:
                cropped = img[pt[1] - w_offset:pt[1] + w - w_offset, pt[0]:pt[0] + h]

            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.show()
            cropped = rotate(cropped, loc_id % len(angles) * -90)
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.show()

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def get_template_locs(img, sizes, angles):
    threshold = 0.7
    locs = []

    templates = ["4_of_clubs.png", "4_of_spades.png", "4_of_hearts.png", "4_of_diamonds.png"]
    for template_card in templates:
        template = il.load_image_cv(assets_parent_path + template_card, is_float32=False)
        tmp = template[90:175, 10:95]

        for size in sizes:
            tmp = resize(tmp, width=size, height=size)

            for angle in angles:
                if angle == 0:
                    res = cv2.matchTemplate(img, tmp, cv2.TM_CCOEFF_NORMED)
                else:
                    res = cv2.matchTemplate(img, rotate(tmp, 90), cv2.TM_CCOEFF_NORMED)
                locs.append(np.where(res >= threshold))

    return locs


# Checks if the identified rectangle is similar as the one before
def check_if_point_exists(current_pt0, current_pt1, pt, same_rectangle_offset):
    for current_pt_id in range(0, len(current_pt0)):
        if current_pt0[current_pt_id] - same_rectangle_offset < pt[0] < current_pt0[
            current_pt_id] + same_rectangle_offset and \
                current_pt1[current_pt_id] - same_rectangle_offset < pt[1] < current_pt1[
                current_pt_id] + same_rectangle_offset:
            return True
    return False


def get_assets():
    asset_images, asset_names = il.load_images(assets_parent_path)
    for img_number in range(0, len(asset_images)):
        # [0:200, 0:100] - the whole rectangle of sign and number
        # [90:175, 10:95] - just the sign of the card
        # [0:90, 0:100] - just the number of the card
        asset_images[img_number] = asset_images[img_number][0:200, 0:100]
        cv2.imshow('image', asset_images[img_number])
        cv2.waitKey()
    return asset_images


def contour_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bigger_cnts = []
    for cnt in cnts:
        approx = cv2.contourArea(cnt)
        if approx > 10000:
            bigger_cnts.append(cnt)
            # x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(gray_img,
            #               (x, y), (x + w, y + h),
            #               (0, 255, 0),
            #               2)

    cv2.drawContours(thresh_img, bigger_cnts, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', thresh_img)
    cv2.waitKey(0)
    d = pytesseract.image_to_string(thresh_img, config="--psm 10")
    print(d)


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
