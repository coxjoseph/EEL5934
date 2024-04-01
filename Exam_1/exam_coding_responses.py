from pathlib import Path
from typing import Union

import cv2
import numpy as np
import matplotlib.pyplot as plt


def theoretical_q2() -> None:
    image = cv2.imread('assets/image_1.jpg', cv2.IMREAD_GRAYSCALE)

    # Normalized kernels
    kernel_3x3 = np.ones((3, 3), np.float32) / 9
    kernel_10x10 = np.ones((10, 10), np.float32) / 100

    # 2D Filtration
    result_3x3 = cv2.filter2D(image, -1, kernel_3x3)
    result_10x10 = cv2.filter2D(image, -1, kernel_10x10)

    # Plot the original and convolved images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(result_3x3, cmap='gray')
    plt.title('Convolution with 3x3 Kernel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result_10x10, cmap='gray')
    plt.title('Convolution with 10x10 Kernel')
    plt.axis('off')

    plt.show()


def segment_zebrafish_eye(image_path, red_threshold=200, min_contiguous_pixels=100) -> np.ndarray:
    """
    Segment the lens of the zebrafish image located at the inputted path, as outlined in exam Segmentation Challenge 1.
    Also displays a figure depicting the original image, and the segmentation.
    :param min_contiguous_pixels: minimum contiguous pixels for segmentation to be considered a lens
    :param red_threshold: threshold value for red channel to segment on
    :param image_path: Path of zebrafish image
    :return: binary mask of segmented image
    """

    # Read and extract red channel from image
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]  # OpenCV BGR ordering

    # Threshold the red channel and find the edges (called contours)
    # This uses cv2's edge detection method, which is more reliable than simple edge detection
    # filtering, but that can be used too.
    _, red_binary = cv2.threshold(red_channel, red_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(red_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with fewer than min_contiguous_pixels using the contour area function
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contiguous_pixels]

    # Create a mask for the filtered contours and draw over our filter
    mask = np.zeros_like(red_binary)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image to get the segmented image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image


def segment_bacteria_boundaries(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Correct non-uniform contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)

    # Laplacian edge detection
    laplacian = cv2.Laplacian(enhanced_image, cv2.CV_64F, ksize=3)
    laplacian_uint8 = cv2.convertScaleAbs(laplacian)  # typing for thresholding

    # Adaptive thresholding based on mean
    binary_edges = cv2.adaptiveThreshold(laplacian_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

    # Filter out centers
    median_filtered = cv2.medianBlur(binary_edges, 3)

    return median_filtered


def sharpen_image(image):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  50, -1],
                       [-1, -1, -1]])
    # Apply the sharpening filter
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def segment_nuclei(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Adaptive thresholding to segment nuclei, then sharpen to reduce noise
    binary_nuclei = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary_nuclei = sharpen_image(binary_nuclei)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_nuclei, connectivity=8)
    min_size = 400  # Minimum size of nuclei

    # Create mask using components and min_size
    nuclei_mask = np.zeros_like(binary_nuclei)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            nuclei_mask[labels == label] = 255

    return nuclei_mask


if __name__ == '__main__':
    # Q2
    theoretical_q2()

    # Zebrafish
    segmented_zebrafish = segment_zebrafish_eye('assets/Zebrafish_Eye.png', min_contiguous_pixels=500)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(segmented_zebrafish, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Zebrafish Eye')
    plt.axis('off')
    plt.show()

    # Bacteria
    segmented_bacteria = segment_bacteria_boundaries('assets/Bacteria.png')
    plt.figure()
    plt.imshow(segmented_bacteria, cmap='gray')
    plt.title('Segmented Bacteria Boundaries')
    plt.axis('off')
    plt.show()

    # Bonus: nuclei
    segmented_nuclei = segment_nuclei('assets/RespEpi.jpg')
    plt.imshow(segmented_nuclei, cmap='gray')
    plt.title('Segmented Nuclei')
    plt.axis('off')
    plt.show()
