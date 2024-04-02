import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def mask_glomerulus(full_glomerulus_image: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    if not boundary_mask.dtype == bool:
        boundary_mask = boundary_mask.astype(bool)

    boundary_mask = np.repeat(boundary_mask[..., np.newaxis], repeats=3, axis=-1)
    return np.where(boundary_mask, full_glomerulus_image, 0)


def plot_hsv_channels(rgb_image: np.ndarray) -> np.ndarray:
    if not (rgb_image.shape[2] == 3 and len(rgb_image.shape) == 3):
        raise ValueError(f'Image is not RGB with channel dim last, dimensions {rgb_image.shape}')
    normalized = rgb_image / 255
    converted_image = rgb_to_hsv(normalized)
    h_channel, s_channel, v_channel = converted_image[:, :, 0], converted_image[:, :, 1], converted_image[:, :, 2]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(normalized)
    axes[0].set_title('Original RGB')
    axes[0].axis('off')

    for ax, channel, title in zip(axes[1:], [h_channel, s_channel, v_channel], ['Hue', 'Saturation', 'Value']):
        ax.imshow(channel, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()
    return converted_image


def threshold_channel(channel: np.ndarray, otsu: bool = True, threshold: float = None) -> np.ndarray:
    if otsu:
        threshold = threshold_otsu(channel)
    return channel > threshold


def cluster_image_with_kmeans(image: np.ndarray, n_clusters: int = 4, plot_recon: bool = True) -> np.ndarray:
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    clustered_image_labels = kmeans.labels_.reshape(image.shape[0], image.shape[1])
    clustered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            clustered_image[i, j] = kmeans.cluster_centers_[clustered_image_labels[i, j]]

    if plot_recon:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(clustered_image.astype('uint8'))
        plt.title('Clustered Image')
        plt.axis('off')

        plt.show()

    return clustered_image


def inverse_one_hot(one_hot_encoded: np.ndarray) -> np.ndarray:
    return np.argmax(one_hot_encoded, axis=-1)


def calculate_metrics(gt: np.ndarray, predictions: np.ndarray):
    n_classes = gt.shape[-1]
    ground_truth = inverse_one_hot(gt)
    predictions = inverse_one_hot(predictions)

    accuracies = []
    sensitivities = []
    specificities = []

    for i in range(n_classes):
        true_pos = np.sum((ground_truth == i) & (predictions == i))
        true_neg = np.sum((ground_truth != i) & (predictions != i))
        false_pos = np.sum((ground_truth != i) & (predictions == i))
        false_neg = np.sum((ground_truth == i) & (predictions != i))

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    dice_score = f1_score(ground_truth.flatten(), predictions.flatten(), average='weighted')

    return accuracies, sensitivities, specificities, dice_score


def segment_and_evaluate_glom_image(original_image: np.ndarray, mask: np.ndarray, gt: np.ndarray) -> tuple:
    clustered_image = cluster_image_with_kmeans(original_image, n_clusters=4, plot_recon=False)
    masked_image = mask_glomerulus(clustered_image, mask)

    acc, sens, spec, dice = calculate_metrics(gt, masked_image)

    return acc, sens, spec, dice



