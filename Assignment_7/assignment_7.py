import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def apply_2d_fourier_transform(image: np.ndarray, display=False) -> np.ndarray:
    transform = np.fft.fft2(image)
    zero_shifted = np.fft.fftshift(transform)
    cartesian = np.abs(zero_shifted)
    cartesian = 20 * np.log(cartesian)

    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(cartesian, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()

    return zero_shifted


def create_binary_mask(frequency_threshold: float, size: tuple[int, int], filter_type: str = 'low') -> np.ndarray:
    width, height = size
    x, y = np.ogrid[:width, :height]

    center = list(map(lambda i: i//2, size))

    dist = np.sqrt((x - center[0])**2 + (y - center[1] ** 2))
    mask = dist <= frequency_threshold if filter_type == 'low' else dist >= frequency_threshold
    return mask


def filter_(transformed_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, transformed_image, 0)


def plot_thresholded_images(thresholds: list[int], transformed_image: np.ndarray, filter_type: str = 'low') -> None:
    plt.figure(figsize=(25, 25))
    for index, threshold in enumerate(thresholds):
        image_mask = create_binary_mask(threshold, transformed_image.shape, filter_type=filter_type)
        filtered_image = filter_(transformed_image, image_mask)
        filtered_image_spatial = np.fft.ifft2(filtered_image)

        plt.subplot(2, len(thresholds), index+1)
        plt.imshow(20 * np.log(np.abs(filtered_image)), cmap='gray')
        plt.title(f'Fourier transform of {filter_type}-pass image with threshold {threshold}Hz')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2, len(thresholds), index+len(thresholds)+1)
        plt.imshow(filtered_image_spatial, cmap='gray')
        plt.title(f'Low-pass filtered image with threshold {threshold}Hz')
        plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    pod_electron_microscope = Image.open('./assets/Pod_Em.jpg').convert('L')
    pod_image = np.array(pod_electron_microscope)

    transformed_pod = apply_2d_fourier_transform(pod_image, display=True)

    low_pass_thresholds = [10**i for i in range(5)]
    high_pass_thresholds = list(reversed(low_pass_thresholds))

    plot_thresholded_images(low_pass_thresholds, transformed_pod, filter_type='low')
    plot_thresholded_images(high_pass_thresholds, transformed_pod, filter_type='high')

