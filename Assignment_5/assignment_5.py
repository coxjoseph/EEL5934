import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional, Union
from scipy.signal import convolve2d, convolve
from skimage.filters import threshold_otsu
from skimage.morphology import disk, white_tophat


def invert_image(image: Image) -> Image:
    image_array = np.array(image).astype(int)

    inverted_array = 255 - image_array  # values go from -255 to 0
    inverted_image = inverted_array.astype(np.uint8)

    return Image.fromarray(inverted_image)


def display_channel_combinations(image: Image) -> None:
    image_array = np.array(image)

    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    combinations = [
        ("Original", image_array),
        ("Red-Green", np.stack((green_channel, red_channel, blue_channel), axis=2)),
        ("Red-Blue", np.stack((blue_channel, green_channel, red_channel), axis=2)),
        ("Green-Blue", np.stack((red_channel, blue_channel, green_channel), axis=2)),
        ("Green-Red", np.stack((red_channel, blue_channel, green_channel), axis=2)),
        ("Blue-Red", np.stack((green_channel, blue_channel, red_channel), axis=2)),
        ("Blue-Green", np.stack((red_channel, green_channel, blue_channel), axis=2))
    ]

    fig_, axes_ = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (title, combination) in zip(axes_.flatten(), combinations):
        ax.imshow(combination)
        ax.set_title(title)
        ax.axis("off")
    plt.show()


def two_d_conv_example() -> None:
    """
    Python implementation of the provided Two_D_Conv_Example.m file
    """

    x = np.round(10 * np.random.random_sample((5, 5)))
    h = np.random.random_sample((3, 3)) > 0.5

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.title('Random Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(h, cmap='gray')
    plt.axis('off')
    plt.title('Random Convolution Kernel')
    plt.show()

    x_conv_h = convolve2d(x, h)

    # Visualizing convolved image
    plt.figure()
    plt.imshow(x_conv_h, cmap='gray')
    plt.axis('off')
    plt.title('Image Convolved with PSF')
    plt.show()


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


def sobel_edge(image: np.ndarray, kernel_x: Optional[np.ndarray] = None, kernel_y: Optional[np.ndarray] = None,
               display_intermediates: bool = False) -> np.ndarray:
    if kernel_x is None:
        kernel_x = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]])

    if kernel_y is None:
        kernel_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])
    if len(image.shape) == 3:
        kernel_x = np.stack((kernel_x, kernel_x, kernel_x), axis=-1)
        kernel_y = np.stack((kernel_y, kernel_y, kernel_y), axis=-1)
        horizontal = convolve(image, kernel_x, mode='same')
        vertical = convolve(image, kernel_y, mode='same')
    else:
        horizontal = convolve2d(image, kernel_x, mode='same')
        vertical = convolve2d(image, kernel_y, mode='same')

    if display_intermediates and len(image.shape) != 3:
        fig_, axes_, = plt.subplots(1, 2, figsize=(10, 5))
        axes_[0].imshow(horizontal, cmap='gray')
        axes_[0].set_title('Image convolved with x-kernel (horizontal)')
        axes_[0].axis('off')
        axes_[1].imshow(vertical, cmap='gray')
        axes_[1].set_title('Image convolved with y-kernel (vertical)')
        axes_[1].axis('off')
        plt.show()

    edges = np.sqrt(horizontal**2 + vertical**2)
    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.title('Implementation of Sobel Edge Filter')
    plt.show()

    return kernel_y


def im2bw(image: np.ndarray, level=0.5) -> np.ndarray:
    min_intensity, max_intensity = np.min(image), np.max(image)

    threshold = level * (max_intensity - min_intensity)  + min_intensity
    return (image > threshold).astype(np.uint8) * 255


def imbinarize(image: np.ndarray) -> np.ndarray:
    threshold = threshold_otsu(image)
    return (image > threshold).astype(np.uint8) * 255


def imhist(image: np.ndarray) -> None:
    plt.hist(image.flatten(), bins=256, range=(0, 256), color='gray')
    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


def imadjust(image: np.ndarray) -> np.ndarray:
    flat = image.flatten()
    bottom_index = int(0.01 * len(flat))
    top_index = len(flat) - bottom_index

    sorted_image = np.sort(flat)
    lower_limit, upper_limit = sorted_image[bottom_index], sorted_image[top_index]

    adjusted_image = np.where(image < lower_limit, 0,
                              np.where(image > upper_limit, 1, linear_map(image, lower_limit, upper_limit)))

    return adjusted_image


def linear_map(value: Union[int, np.ndarray], lower_limit: int, upper_limit: int) -> int:
    mapped_value = (value - lower_limit) / (upper_limit - lower_limit)
    return mapped_value


def show_single(image: np.ndarray, grayscale=True) -> None:
    cmap = 'gray' if grayscale else None
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Question 1a:
    covid_image = Image.open('assets/SarsCoV2.jpg')
    covid_inverted = invert_image(covid_image)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(covid_image)
    axes[0].set_title('Original SARS CoV2 Image')
    axes[0].axis('off')
    axes[1].imshow(covid_inverted)
    axes[1].set_title('Inverted SARS CoV2 Image')
    axes[1].axis('off')
    plt.show()

    # Question 1b:
    covid_if_image = Image.open('assets/SarsCoV2_IF.jpg')
    display_channel_combinations(covid_if_image)

    # Question 2a:
    two_d_conv_example()

    # Question 3a
    circle = one_sphere(radius=25, intensity=15, center=(128, 128))

    # Question 3b and c
    sobel_edge(circle, display_intermediates=True)

    # Question 3d
    cell_image = np.array(Image.open('assets/cell.jpg').convert('L'))
    sobel_edge(cell_image)

    # Question 5.1.a
    tophat_image = np.array(Image.open('assets/tophat.png').convert('L'))
    imhist(tophat_image)

    # Question 5.1.c and d
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(im2bw(tophat_image), cmap='gray')
    axes[0].set_title('Segmentation from im2bw')
    axes[0].axis('off')
    axes[1].imshow(imbinarize(tophat_image), cmap='gray')
    axes[1].set_title('Segmentation from imbinarize')
    axes[1].axis('off')
    plt.show()

    # Question 5.2.c
    tophat_filtered = white_tophat(tophat_image, disk(12))  # Radius of 12 given in problem
    show_single(tophat_filtered)

    # Question 5.2.d
    imhist(tophat_filtered)

    # Question 5.2.e
    tophat_adjusted = imadjust(tophat_filtered)
    tophat_adjusted_binarized = imbinarize(tophat_adjusted)
    show_single(tophat_adjusted_binarized)

    # Question 5.3.a
    kernel = np.array([[2, 1, 0],
                       [1, 0, -1],
                       [0, -1, 2]])

    tophat_convolved = convolve2d(tophat_adjusted_binarized, kernel)
    show_single(tophat_convolved)

    # Question 5.3.b
    opposite_kernel = -1 * kernel
    opposite_convolved = convolve2d(tophat_adjusted_binarized, opposite_kernel)
    show_single(opposite_convolved)
