import numpy as np
import matplotlib.pyplot as plt


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

    dist = np.sqrt((x - center[0])**2 + (y - center[1]) ** 2)
    mask = dist <= frequency_threshold if filter_type == 'low' else dist >= frequency_threshold
    return mask


def filter_(transformed_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, transformed_image, 0)


def plot_thresholded_images(thresholds: list[int], transformed_image: np.ndarray, filter_type: str = 'low') -> None:
    plt.figure(figsize=(15, 10))
    for index, threshold in enumerate(thresholds):
        image_mask = create_binary_mask(threshold, transformed_image.shape, filter_type=filter_type)
        filtered_image = filter_(transformed_image, image_mask)
        unshifted_image = np.fft.ifftshift(filtered_image)
        filtered_image_spatial = np.fft.ifft2(unshifted_image)

        plt.subplot(2, len(thresholds), index+1)
        plt.imshow(20 * np.log(np.abs(filtered_image)), cmap='gray')
        plt.title(f'{filter_type}-pass transform \nThreshold {threshold}Hz')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2, len(thresholds), index+len(thresholds)+1)
        plt.imshow(np.abs(filtered_image_spatial), cmap='gray')
        plt.title(f'{filter_type}-pass image \nThreshold {threshold}Hz')
        plt.xticks([]), plt.yticks([])
    plt.show()
