import numpy as np
import random
import math
import cv2
import pandas as pd
from skimage.util import random_noise
from skimage.measure import regionprops_table
from skimage.morphology import label
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.optimizers import Adam
import matplotlib.pyplot as plt


def create_motion_blur_kernel() -> np.ndarray:
    size = random.randint(1, 100)
    direction = random.randint(0, 145)

    direction = math.radians(direction)

    matrix_size = 2 * size
    center = matrix_size // 2

    kernel = np.zeros((matrix_size, matrix_size))

    x0 = int(center + size * math.cos(direction) / 2)
    y0 = int(center + size * math.sin(direction) / 2)
    x1 = int(center - size * math.cos(direction) / 2)
    y1 = int(center - size * math.sin(direction) / 2)

    kernel = cv2.line(kernel, (x0, y0), (x1, y1), (255, ), 1)
    return kernel


def process_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    blurred_image = cv2.filter2D(image, -1, kernel)

    threshold = random.randint(100, 200)
    _, binarized_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)

    noisy_image = random_noise(binarized_image, mode='gaussian', mean=20, var=49)
    final_image = random_noise(noisy_image, mode='s&p', amount=0.05)
    final_image = cv2.resize(final_image, (50, 50), interpolation=cv2.INTER_AREA)
    final_image *= 255
    final_image = final_image.astype(np.uint16)
    return final_image


def region_properties(image_array: np.ndarray) -> np.ndarray:
    properties_array = []
    for individual_image in image_array:
        _, binarized_image = cv2.threshold(individual_image, 0, 255, cv2.THRESH_OTSU)

        labeled_image = label(binarized_image)
        properties = regionprops_table(labeled_image,
                                       properties=('area', 'eccentricity', 'solidity', 'perimeter', 'extent'))
        properties_array.append(pd.DataFrame(properties).to_numpy())
    print(properties_array)
    print(properties_array[0].shape)

    return np.array(properties_array)


def train_mlp(data: np.ndarray, labels: np.ndarray, learning_rate: float = 5e-15,
              n_epochs: int = 1e4, hidden_layer_size : int = 10, plot: bool = False) -> float:
    num_classes = labels.shape[1]

    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)  # Flatten each image to a 1D vector

    data = data.astype('float32') / 255.0

    model = Sequential([
        Flatten(input_shape=(data.shape[1], )),
        Dense(hidden_layer_size, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(data, labels, epochs=n_epochs, verbose=1)
    final_accuracy = history.history['accuracy'][-1]
    print(f'Final accuracy: {final_accuracy}')

    if plot:
        epochs = [i for i in range(n_epochs)]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.history['loss'], label='Loss')
        plt.title('Loss by Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.history['accuracy'], label='Accuracy')
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.show()

    return final_accuracy


if __name__ == '__main__':
    # Q1
    circle = cv2.imread('./assets/Circle_bin.png', cv2.IMREAD_GRAYSCALE)
    star = cv2.imread('./assets/Star_bin.png', cv2.IMREAD_GRAYSCALE)
    modified_circles = []
    modified_stars = []
    for _ in range(100):
        motion_blur = create_motion_blur_kernel()
        modified_circles.append(process_image(circle, kernel=motion_blur))
        modified_stars.append(process_image(star, kernel=motion_blur))

    circles = np.array(modified_circles)
    stars = np.array(modified_stars)

    # Q2:
    circle_pixel_features = circles.reshape((100, 50*50))
    star_pixel_features = stars.reshape((100, 50*50))
    pixel_features = np.vstack((circle_pixel_features, star_pixel_features))

    circle_binarized_features = region_properties(circles)
    star_binarized_features = region_properties(stars)
    binarized_features = np.vstack((circle_binarized_features, star_binarized_features))

    image_labels = [[1, 0]] * 100
    image_labels.extend([[0, 1]] * 100)
    image_labels = np.array(image_labels)


    # Q3:
    default_pixel_accuracy = train_mlp(pixel_features, image_labels, plot=True)
    better_pixel_accuracy = train_mlp(pixel_features, image_labels, hidden_layer_size=15, learning_rate=1e-5, plot=True)

    default_feature_accuracy = train_mlp(binarized_features, image_labels, plot=True)
    better_feature_accuracy = train_mlp(binarized_features, image_labels, hidden_layer_size=15, learning_rate=1e-5,
                                        plot=True)
