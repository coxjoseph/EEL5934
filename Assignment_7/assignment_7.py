import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from question_1 import apply_2d_fourier_transform, plot_thresholded_images
from question_2 import plot_hsv_channels, threshold_channel, cluster_image_with_kmeans, segment_and_evaluate_glom_image
from question_3 import one_sphere
from question_4 import load_image, evaluate_segmentation, create_segmentation_label

from skimage.filters import butterworth


if __name__ == '__main__':
    pod_electron_microscope = Image.open('./assets/Pod_Em.jpg').convert('L')
    pod_image = np.array(pod_electron_microscope)

    transformed_pod = apply_2d_fourier_transform(pod_image, display=True)

    low_pass_thresholds = [10, 50, 100, 500, 1000]
    high_pass_thresholds = list(reversed(low_pass_thresholds))

    plot_thresholded_images(low_pass_thresholds, transformed_pod, filter_type='low')
    plot_thresholded_images(high_pass_thresholds, transformed_pod, filter_type='high')

    # Q2:
    # noinspection PyTypeChecker
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

    for filename in os.listdir('./assets/Glomeruli_Images'):
        identifier = filename.split('_raw.png')[0]
        image = np.array(Image.open(f'./assets/Glomeruli_Images/{filename}'))
        mask = np.array(Image.open(f'./assets/Glomeruli Boundary Masks/{identifier}_mask.png'))
        gt = np.array(Image.open(f'./assets/Glomeruli_GT_Masks/{identifier}_comp.png'))

        result = segment_and_evaluate_glom_image(image, mask, gt)
        print(f'--------------- {identifier} ---------------')
        for item, label in zip(result, ['Accuracy', 'Sensitivity', 'Specificity', 'Dice']):
            print(f'{label}: {item}')

    circle_image = one_sphere(15, 150, (40, 40))
    transform_image = apply_2d_fourier_transform(circle_image, display=True)
    high_passed = butterworth(circle_image, high_pass=True)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(circle_image, cmap='gray')
    plt.title('Original circle')
    plt.subplot(1, 2, 2)
    plt.xticks([]), plt.yticks([])
    plt.imshow(high_passed, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # Q4
    stripes = load_image('./assets/pattern_seg.png')
    segmentation = create_segmentation_label(stripes)
    ground_truth = load_image('./assets/ground_truth_seg.png')
    result = evaluate_segmentation(segmentation, ground_truth)
    print('----------------------------')
    for item, label in zip(result, ['Accuracy', 'Sensitivity', 'Specificity']):
        print(f'{label}: {item}')

