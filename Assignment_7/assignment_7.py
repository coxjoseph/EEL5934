import os

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import numpy as np
from PIL import Image

from question_1 import apply_2d_fourier_transform, plot_thresholded_images
from question_2 import plot_hsv_channels, threshold_channel, cluster_image_with_kmeans, segment_and_evaluate_glom_image


if __name__ == '__main__':
    pod_electron_microscope = Image.open('./assets/Pod_Em.jpg').convert('L')
    pod_image = np.array(pod_electron_microscope)

    transformed_pod = apply_2d_fourier_transform(pod_image, display=True)

    low_pass_thresholds = [10, 50, 100, 500, 1000]
    high_pass_thresholds = list(reversed(low_pass_thresholds))

    plot_thresholded_images(low_pass_thresholds, transformed_pod, filter_type='low')
    plot_thresholded_images(high_pass_thresholds, transformed_pod, filter_type='high')

    # Q2:
    example_glom_image = np.array(Image.open('./assets/Glomeruli_Images/XY01_IU-21-015F2_raw.png'))
    hsv_example = plot_hsv_channels(example_glom_image)

    sat = hsv_example[:, :, 1]
    threshold = 0.47
    thresholded_sat = threshold_channel(sat, otsu=False, threshold=threshold)

    plt.figure()
    plt.imshow(thresholded_sat.astype(int), cmap='gray')
    plt.title(f'Nuclei segmentation with threshold {threshold}')
    plt.axis('off')
    plt.show()

    hue = hsv_example[:, :, 0]
    thresholded_hue = threshold_channel(hue, otsu=True)
    inverted_nuclei = np.logical_not(thresholded_sat)
    pas_segmentation = np.logical_and(thresholded_hue, inverted_nuclei)

    plt.figure()
    plt.imshow(pas_segmentation.astype(int), cmap='gray')
    plt.title(f'PAS segmentation with threshold {threshold}')
    plt.axis('off')
    plt.show()

    inverted_pas = np.logical_not(pas_segmentation)
    lumen = np.logical_and(inverted_nuclei, inverted_pas)

    plt.figure()
    plt.imshow(lumen.astype(int), cmap='gray')
    plt.title(f'Lumen segmentation with threshold {threshold}')
    plt.axis('off')
    plt.show()

    # K Means
    hsv_example = (hsv_example * 255).astype(int)
    kmeans_example = cluster_image_with_kmeans(example_glom_image, plot_recon=True)
    hsv_kmeans = cluster_image_with_kmeans(hsv_example, plot_recon=True)

    for image, mask, gt in zip(os.listdir('./assets/Glomeruli_Images'),
                               os.listdir('./assets/Glomeruli Boundary Masks'),
                               os.listdir('./assets/Glomeruli_GT_Masks')):
        image_array = np.array(Image.open(f'./assets/Glomeruli_Images/{image}'))
        mask_array = np.array(Image.open(f'./assets/Glomeruli Boundary Masks/{mask}'))
        gt_array = np.array(Image.open(f'./assets/Glomeruli_GT_Masks/{gt}'))

        result = segment_and_evaluate_glom_image(image_array, mask_array, gt_array)
        print(f'--------------- {image} ---------------')
        for item in result:
            print(item)

