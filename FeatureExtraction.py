import numpy as np
import math
from scipy import signal


def Gabor_filter(x, y, f, delta_x, delta_y):
    """
        calculate the value

        Args:
            x: x coordinates of the pixel
            y: y coordinates of the pixel
            f: frequency of the sinusoidal function
            delta_x: constant
            delta_y: constant

        Returns:
           result: calculated kernal value
    """
    # transform the equations in the paper into code
    M = np.cos(2 * np.pi * f * math.sqrt(x ** 2 + y ** 2))
    constant = 2 * np.pi * delta_x * delta_y
    exp = np.exp(-1 / 2 * (x ** 2 / delta_x ** 2 + y ** 2 / delta_y ** 2))
    result = (1 / constant) * exp * M
    return result


def filter_kernal(f, delta_x, delta_y, size):
    """
        calculate the final filter

        Args:
            f: frequency of the sinusoidal function
            delta_x: constant
            delta_y: constant
            size: filter kernal size

        Returns:
           result: a size*size grid of filter
    """
    kernal = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernal[i, j] = Gabor_filter((-int(size / 2) + j), (-int(size / 2) + i), f, delta_x, delta_y)
    return kernal


def featureVector(img, threshold_x=48, block=8, f_1=1 / 3, f_2=1 / 3):
    """
        transfer the image to a feature vector

        Args:
            img: image after enhancement
            threshold_x: threshold for roi of the image
            block: size of the kernal filter
            f1/f2: frequency of the sinusoidal function for the two different channels

        Returns:
           feature_vector
    """
    filter_kernal_1 = filter_kernal(f_1, 3, 1.5, block)
    filter_kernal_2 = filter_kernal(f_2, 4.5, 1.5, block)

    roi = img[:threshold_x, :]
    # print(len(roi))

    F1 = signal.convolve2d(roi, filter_kernal_1, mode='same')
    F2 = signal.convolve2d(roi, filter_kernal_2, mode='same')

    feature_vector = []

    # calculate every block value according to the paper
    # since the size not equal to one, we set the step size equals block size.
    for x in range(0, len(roi), block):
        for y in range(0, len(roi[x]), block):
            F1_tmp = F1[x:x + block, y:y + block]
            F2_tmp = F2[x:x + block, y:y + block]

            # channel 1
            m_1 = np.mean(np.absolute(F1_tmp))
            feature_vector.append(m_1)
            sigma_1 = np.mean(np.absolute((np.absolute(F1_tmp)) - m_1))
            feature_vector.append(sigma_1)

            # channel 2
            m_2 = np.mean(np.absolute(F2_tmp))
            feature_vector.append(m_2)
            sigma_2 = np.mean(np.absolute((np.absolute(F2_tmp)) - m_2))
            feature_vector.append(sigma_2)

    return feature_vector
