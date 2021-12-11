import numpy as np
import cv2
from scipy.spatial import distance


def localization(image):
    """
        Localize where the pupil and iris lie

        Args:
            image: path of a image

        Returns:
           x_center2, y_center2, pupil_radius, iris_radius, img_result
    """

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove noise from image
    img_grey = img.copy()
    img_grey = cv2.bilateralFilter(img_grey, 9, 75, 75)

    FILESIZE = 60
    IMAGE_HEIGHT, IMAGE_WIDTH, COLOR = image.shape

    # step1
    col_mean0 = np.mean(img_grey, 0)
    row_mean0 = np.mean(img_grey, 1)

    x_center0 = np.argmin(col_mean0)
    y_center0 = np.argmin(row_mean0)

    if x_center0-FILESIZE < 0 or x_center0+FILESIZE >= IMAGE_WIDTH:
        x_center0 = int(IMAGE_WIDTH/2)
    if y_center0 - FILESIZE < 0 or y_center0 + FILESIZE >= IMAGE_HEIGHT:
        y_center0 = int(IMAGE_HEIGHT/2)

    # step 2
    img_chopped1 = img_grey[y_center0 - FILESIZE:y_center0 + FILESIZE, x_center0 - FILESIZE:x_center0 + FILESIZE]

    col_mean1 = np.mean(img_chopped1, 0)
    row_mean1 = np.mean(img_chopped1, 1)

    x_center1 = x_center0 - FILESIZE + np.argmin(col_mean1)
    y_center1 = y_center0 - FILESIZE + np.argmin(row_mean1)

    if x_center1-FILESIZE < 0 or x_center1+FILESIZE >= IMAGE_WIDTH:
        x_center1 = x_center0
    if y_center1 - FILESIZE < 0 or y_center1 + FILESIZE >= IMAGE_HEIGHT:
        y_center1 = y_center0

    img_chopped2 = img_grey[y_center1 - FILESIZE:y_center1 + FILESIZE, x_center1 - FILESIZE:x_center1 + FILESIZE]

    col_mean2 = np.mean(img_chopped2, 0)
    row_mean2 = np.mean(img_chopped2, 1)

    x_center2 = x_center1 - FILESIZE + np.argmin(col_mean2)
    y_center2 = y_center1 - FILESIZE + np.argmin(row_mean2)

    if x_center2 - FILESIZE < 0 or x_center2 + FILESIZE >= IMAGE_WIDTH:
        x_center2 = x_center1
    if y_center2 - FILESIZE < 0 or y_center2 + FILESIZE >= IMAGE_HEIGHT:
        y_center2 = y_center1

    img_chopped3 = img_grey[y_center2 - FILESIZE:y_center2 + FILESIZE, x_center2 - FILESIZE:x_center2 + FILESIZE]

    center_pupil = [x_center2, y_center2]

    # step 3

    # threshold value is 75, max value 255
    _, image_binary = cv2.threshold(img_grey, 75, 255, cv2.THRESH_BINARY)
    # edge pixels above the 220 are considered and edge pixels below the threshold 100 are discarded
    image_edge = cv2.Canny(image_binary, 100, 220)

    circles = cv2.HoughCircles(image_edge, cv2.HOUGH_GRADIENT, 10, 100)  # get bunch of circles
    circles = np.squeeze(circles, axis=0)

    min_dist = None

    for crl in circles:

        dist = distance.euclidean(center_pupil, [crl[0], crl[1]])

        if not min_dist or dist < min_dist:
            min_dist = dist
            min_circle = crl

    # inner boundary
    # min_circle[0] -- x coordinate in circle
    # min_circle[1] -- y coordinate in circle
    pupil_radius = int(min_circle[2])

    # inner boundary
    img_result = img.copy()
    cv2.circle(img=img_result, center=center_pupil, radius=pupil_radius, color=(255, 0, 0), thickness=2)

    # outer boundary
    iris_radius = pupil_radius + 53
    cv2.circle(img=img_result, center=center_pupil, radius=pupil_radius + 63, color=(255, 0, 0), thickness=2)

    return x_center2, y_center2, pupil_radius, iris_radius, img_result

