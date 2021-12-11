import cv2
import numpy as np
import matplotlib.pyplot as plt


def enhancement(normalized_img, SIZE=32):
    """
        enhancing the normalized iris

        Args:
            normalized_img: image after normalization
            SIZE: constant - refers to the size of region we will enchance, the default value is 32

        Returns: image after enhancement
    """
    img_gray = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)

    img_histEqualization = np.zeros([img_gray.shape[0], img_gray.shape[1]])
    img_count = np.zeros([img_gray.shape[0], img_gray.shape[1]])

    #
    MOVEMENT = SIZE // 8

    x = SIZE//2

    # enhance each 32*32 region
    for i in range(img_gray.shape[1] // MOVEMENT):

        y = SIZE//2

        for j in range(img_gray.shape[0] // MOVEMENT):

            if y < img_gray.shape[0]:
                img_histEqualization[y - SIZE//2: y + SIZE//2, x - SIZE//2: x + SIZE//2] = cv2.equalizeHist(
                    img_gray[y - SIZE//2: y + SIZE//2, x - SIZE//2: x + SIZE//2])
                # img_count[y - SIZE//2: y + SIZE//2, x - SIZE//2: x + SIZE//2] += 1
                # img_histEqualization[y - SIZE//2: y + SIZE//2, x - SIZE//2: x + SIZE//2] += cv2.equalizeHist(
                #     img_gray[y - SIZE//2: y + SIZE//2, x - SIZE//2: x + SIZE//2])


                y += MOVEMENT

        x += MOVEMENT
    # print(img_count[32][24])
    # for i in range(len(img_histEqualization)):
    #     for j in range(len(img_histEqualization[i])):
    #         img_histEqualization[i][j] = img_histEqualization[i][j]/img_count[i][j]
    # plt.imshow(img_histEqualization)
    return img_histEqualization