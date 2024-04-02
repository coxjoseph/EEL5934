from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def load_image(path: str) -> np.ndarray:
    image = Image.open(path)
    image = image.convert('L')
    return np.array(image)


def create_segmentation_label(image: np.ndarray) -> np.ndarray:

    # Vertical and horizontal Sobel filters
    sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # Apply filters
    vertical_response = convolve(image, sobel_vertical)
    horizontal_response = convolve(image, sobel_horizontal)

    # Generate segmentation label based on the dominant response
    segmentation_label = np.where(np.abs(vertical_response) > np.abs(horizontal_response), 1, 0)

    return segmentation_label


def evaluate_segmentation(segmentation: np.ndarray, ground_truth: np.ndarray) -> tuple:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(segmentation, cmap='gray')
    plt.title('Stripe segmentation')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth Label')
    plt.xticks([]), plt.yticks([])
    plt.show()

    n_classes = 2
    accuracies = []
    sensitivities = []
    specificities = []

    # Calculate metrics for each class
    for i in range(n_classes):
        true_pos = np.sum((ground_truth == i) & (segmentation == i))
        true_neg = np.sum((ground_truth != i) & (segmentation != i))
        false_pos = np.sum((ground_truth != i) & (segmentation == i))
        false_neg = np.sum((ground_truth == i) & (segmentation != i))

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return accuracies, sensitivities, specificities
