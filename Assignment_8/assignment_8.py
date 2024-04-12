import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def segment_dic(dic_image: np.ndarray, display: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(dic_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 125)

    kernel = np.ones((13, 13), np.uint8)

    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_edges = np.zeros_like(gray)
    cv2.drawContours(filled_edges, contours, -1, 255, thickness=cv2.FILLED)

    if display:
        plt.figure()
        plt.imshow(filled_edges, cmap='gray')
        plt.show()
    return filled_edges


def segment_fluorescence(fluorescent_image: np.ndarray, display: bool = True) -> np.ndarray:
    red_channel = fluorescent_image[:, :, 2]
    _, binary = cv2.threshold(red_channel, 127, 255, cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((37, 37), np.uint8)
    closed_edges = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_cells = np.zeros_like(closed_edges)
    cv2.drawContours(filled_cells, contours, -1, 255, thickness=cv2.FILLED)

    if display:
        plt.figure()
        plt.imshow(filled_cells, cmap='gray')
        plt.show()
    return filled_cells


def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 255]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1_score
    }


if __name__ == '__main__':
    dic = cv2.imread('assets/SST-ANT_DIC.jpg')
    fluor = cv2.imread('assets/SST-ANT_fluor.jpg')
    gt = cv2.imread('assets/SST-ANT_GT.jpg')
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    dic_pred = segment_dic(dic, display=True)
    fluor_pred = segment_fluorescence(fluor, display=True)

    dic_metrics = calculate_metrics(gt, dic_pred)
    fluor_metrics = calculate_metrics(gt, fluor_pred)

    print(dic_metrics)
    print(fluor_metrics)
