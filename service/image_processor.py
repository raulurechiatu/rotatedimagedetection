import cv2

#NOT USED YET


def contour_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for cnt in cnts:
        approx = cv2.contourArea(cnt)
        # print(approx)

    cv2.imshow('image', img)
    cv2.imshow('Binary', thresh_img)
    cv2.waitKey()


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

