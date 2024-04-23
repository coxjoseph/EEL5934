import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def segment_podocytes(image: np.ndarray, threshold_percentage: float = 0.2, display: bool = False) -> np.ndarray:
    image[image == 0] = 255  # Switch background color

    threshold = np.max(image) * threshold_percentage
    _, binary_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY_INV)
    if display:
        cv2.imshow('Binary Image', binary_image)
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kernel = np.ones((5, 5), np.uint8)

    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    if display:
        cv2.imshow('Segmented Image', binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary_image


def feature_extraction(image: np.ndarray) -> dict:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    circularities = []
    solidities = []
    perimeters = []

    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

        perimeter = cv2.arcLength(contour, True)
        perimeters.append(perimeter)

        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularities.append(circularity)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        solidities.append(solidity)

    areas = np.array(areas)
    circularities = np.array(circularities)
    solidities = np.array(solidities)
    perimeters = np.array(perimeters)

    median_area = np.median(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)

    median_circularity = np.median(circularities)
    min_circularity = np.min(circularities)
    max_circularity = np.max(circularities)

    median_solidity = np.median(solidities)
    min_solidity = np.min(solidities)
    max_solidity = np.max(solidities)

    median_perimeter = np.median(perimeters)
    min_perimeter = np.min(perimeters)
    max_perimeter = np.max(perimeters)

    return {
        'median_area': median_area,
        'min_area': min_area,
        'max_area': max_area,
        'median_circularity': median_circularity,
        'min_circularity': min_circularity,
        'max_circularity': max_circularity,
        'median_solidity': median_solidity,
        'min_solidity': min_solidity,
        'max_solidity': max_solidity,
        'median_perimeter': median_perimeter,
        'min_perimeter': min_perimeter,
        'max_perimeter': max_perimeter
    }


if __name__ == '__main__':
    counter = 0
    features = pd.DataFrame(columns=['image_name', 'median_area', 'min_area', 'max_area',
                                     'median_circularity', 'min_circularity', 'max_circularity',
                                     'median_perimeter', 'min_perimeter', 'max_perimeter', 'label'])

    labels = pd.read_csv('./assets/Glomeruli_Labels.csv', header=0)

    for glom_path in os.listdir('./assets/P57_Glomeruli'):
        if glom_path.endswith('.png'):
            filename = f'./assets/P57_Glomeruli/{glom_path}'
            plot = counter < 4
            grayscale_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            binary_podocytes = segment_podocytes(grayscale_image, display=plot)
            counter += 1

            image_features = feature_extraction(binary_podocytes)
            image_features['label'] = labels.loc[labels['ImageName'] == glom_path, 'Label'].iloc[0]
            image_features['image_name'] = glom_path
            image_features = pd.DataFrame(image_features, index=[0])
            features = pd.concat([features, image_features], ignore_index=True)

    features_list = [col for col in features.columns if col != 'label']

    for feature in features_list:
        sns.boxplot(x='label', y=feature, data=features)
        plt.title(f'Boxplot of {feature} by Label')
        plt.xlabel('Label')
        plt.ylabel(feature)
        plt.show()
