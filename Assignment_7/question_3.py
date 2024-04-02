import numpy as np
from typing import Tuple


def one_sphere(radius: int, intensity: int, center: Tuple[int, int]) -> np.ndarray:
    """
    My guess as to what one_sphere.m does based on the question. Create a circle with given center, radius, and
    intensity on a 256x256 image grid.

    :param radius: Radius of the circle, in pixels
    :param intensity: Intensity of the circle, between 0 and 255
    :param center: Center of the circle, 0-indexed.
    :return: Array with shape (256, 256), a one-channel (grayscale) image with the circle on it.
    """

    if not (0 <= intensity <= 255):
        raise ValueError('Intensity must be between 0 and 255')

    x, y = np.meshgrid(np.arange(256), np.arange(256))

    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)  # Array of euclidean distance from center
    circle_image = np.where(dist <= radius, intensity, 0)  # Numpy ternary operator
    return circle_image
