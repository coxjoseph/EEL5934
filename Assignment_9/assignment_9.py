import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_gaussian_noise(image: np.ndarray, mean: float = 0., standard_dev: float = 10.) -> np.ndarray:
    noise = np.random.normal(mean, standard_dev, image.shape)
    noisy_image = image + noise

    noisy_image = np.clip(noisy_image, 0, 255).astype(image.dtype)
    return noisy_image


def sobel_edge_filter(image: np.ndarray, display: bool = True, title: str = None) -> np.ndarray:
    """
    This should be the implementation of MATLAB's edge() method. Image should be grayscale
    """

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobel_edges = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    sobel_edges = np.uint8(255 * sobel_edges / np.max(sobel_edges))
    _, sobel_edges = cv2.threshold(sobel_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if display:
        plt.figure()
        plt.imshow(sobel_edges, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(title)
        plt.show()

    return sobel_edges


def low_pass_filter(image: np.ndarray, filter_size: int = 3) -> np.ndarray:
    """
    This is just an average filter, which is a low-pass filter itself
    """

    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size ** 2)
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image


if __name__ == '__main__':
    butterfly_image = cv2.imread('assets/ButterflyWing.PNG')
    butterfly = cv2.cvtColor(butterfly_image, cv2.COLOR_BGR2GRAY)

    mean = 10
    standard = round((0.5 * 255) ** 0.5, 0)
    noisy_butterfly = add_gaussian_noise(butterfly, mean, standard)
    plt.figure()
    plt.imshow(noisy_butterfly, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    edge_butterfly = sobel_edge_filter(butterfly, display=True, title='no noise')
    edge_noise = sobel_edge_filter(noisy_butterfly, display=True, title='noise')

    sizes = [i for i in range(3, 15, 2)]
    plt.figure(figsize=(20, 6))
    for i, kernel_size in enumerate(sizes):
        filtered_butterfly = low_pass_filter(edge_butterfly, kernel_size)
        plt.subplot(1, len(sizes), i+1)
        plt.imshow(filtered_butterfly, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(f'Filter size {kernel_size}')
    plt.show()
