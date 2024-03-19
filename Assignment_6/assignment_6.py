import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, medfilt2d, wiener
from scipy import ndimage


def object_generation(size: int, center: tuple, radius: int, signal: int) -> np.ndarray:
    original_image = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)

            if dist <= radius:
                original_image[i, j] = signal

    return original_image


def salt_pepper_noise(original_image, salt_freq: float = 0.05, pepper_freq: float = 0.05) -> np.ndarray:
    size = original_image.shape[0]
    num_salt = np.ceil(size ** 2 * salt_freq)
    num_pepper = np.ceil(size ** 2 * pepper_freq)

    salt_coords = [np.random.randint(0, size, int(num_salt)),
                   np.random.randint(0, size, int(num_salt))]
    pepper_coords = [np.random.randint(0, size, int(num_pepper)),
                     np.random.randint(0, size, int(num_pepper))]

    noisy_image = np.copy(original_image)  # Don't modify in-place

    xs, ys = salt_coords[0], salt_coords[1]
    for x, y in zip(xs, ys):
        noisy_image[x, y] = 255

    xs, ys = pepper_coords[0], pepper_coords[1]
    for x, y in zip(xs, ys):
        noisy_image[x, y] = 0

    return noisy_image


def gaussian_noise(original_image: np.ndarray, mean: float = 0.0, std_dev: float = 100.0) -> np.ndarray:
    noise = np.random.normal(mean, std_dev, original_image.shape)

    noisy_image = np.copy(original_image)  # Don't modify in-place
    noisy_image += noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def generate_noisy_image(size: int, center: tuple, radius: int, signal: int) -> np.ndarray:
    image = object_generation(size, center, radius, signal)
    peppered = salt_pepper_noise(image)
    noised = gaussian_noise(peppered, std_dev=5)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(peppered, cmap='gray')
    plt.title('Salt & Pepper')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(noised, cmap='gray')
    plt.title('Object with Noise')
    plt.axis('off')
    plt.show()

    return noised


def apply_filters(image: np.ndarray, filter_1: np.ndarray, filter_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original_order = convolve2d(convolve2d(image, filter_1), filter_2)
    flipped_order = convolve2d(convolve2d(image, filter_2), filter_1)

    plot_flipped_filters(original_order, flipped_order)
    return original_order, flipped_order


def plot_flipped_filters(original_order: np.ndarray, flipped_order: np.ndarray) -> None:
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_order, cmap='gray')
    plt.title('Original Order')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(flipped_order, cmap='gray')
    plt.title('Reversed Order')
    plt.axis('off')
    plt.show()


def clean_cell_image(image: np.ndarray) -> np.ndarray:
    median_filtered = medfilt2d(image, kernel_size=5)
    wiener_filtered = wiener(median_filtered, 5)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original (noisy) image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(wiener_filtered, cmap='gray')
    plt.title('Restored image')
    plt.axis('off')
    plt.show()
    return wiener_filtered


if __name__ == '__main__':
    source_image = generate_noisy_image(size=200, center=(100, 100), radius=25, signal=150)

    q1a_filter_1 = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

    q1a_filter_2 = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

    apply_filters(source_image, q1a_filter_1, q1a_filter_2)

    q1b_filter_1 = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

    og_order = medfilt2d(convolve2d(source_image, q1b_filter_1), kernel_size=3)
    rev_order = convolve2d(medfilt2d(source_image, kernel_size=3), q1b_filter_1)
    plot_flipped_filters(og_order, rev_order)

    cell_image = plt.imread('./assets/blurry_cells.png')
    restored_image = clean_cell_image(cell_image)
