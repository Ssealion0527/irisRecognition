
import numpy as np

from IrisLocalization import localization


def normalization(image):
    """
        mapping the iris from Cartesian coordinates to polar coordinates;

        Args:
            image: path of a image

        Returns: image after normalization
    """

    center_x, center_y, inner_radius, outer_radius, image = localization(image)

    I_n = np.zeros((64, 512, 3))  # M*N is 64*512

    for Y in range(64):
        for X in range(512):
            theta = 2 * np.pi * X / 512

            # inner boundary coordinate: x_p, y_p
            x_p = center_x + np.cos(theta) * inner_radius
            y_p = center_y + np.sin(theta) * inner_radius

            x_p = np.around(x_p)
            y_p = np.around(y_p)

            # outer boundary coordinate: x_i, y_i
            x_i = center_x + np.cos(theta) * outer_radius
            y_i = center_y + np.sin(theta) * outer_radius

            x_i = np.around(x_i)
            y_i = np.around(y_i)

            # original coordinate
            x = x_p + (x_i - x_p) * (Y / 64)
            y = y_p + (y_i - y_p) * (Y / 64)

            # there are case where the coordinate is out of boundary.
            # Put them back to where boundary lies
            if x >= 320:
                x = 319
            if y >= 280:
                y = 279

            I_n[Y][X] = image[int(y)][int(x)]

    I_n = I_n.astype(np.uint8)

    return I_n
