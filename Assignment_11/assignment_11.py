import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture


def process_image(path: str, plot: bool = False,
                  colorspace: Optional[int] = None) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    image = cv2.imread(path)
    if colorspace is not None:
        image = cv2.cvtColor(image, code=colorspace)
    m, n, c = image.shape

    image_vector = image.reshape((m * n), c)

    if plot:
        plot_3d_scatter(image_vector, title='3D Scatter Plot of pixel colors', filename='3d_scatter.png')
    return image_vector, (m, n, c)


def fit_gmm_dist(data: np.ndarray, n_clusters: int, max_iterations: int,
                 display: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    gmm = GaussianMixture(n_components=n_clusters, max_iter=max_iterations,
                          random_state=3621, covariance_type='diag', init_params='k-means++')
    labels = gmm.fit_predict(data)

    if display:
        plot_3d_scatter(data, title=f'GMM Clustering with {n_clusters} clusters', filename=f'gmm_{n_clusters}.png',
                        labels=labels)

    log_likelihood = gmm.score(data)
    print(f'Per-sample average log-likelihood of GMM ({n_clusters} clusters): {log_likelihood}')

    return labels, gmm.means_


def plot_3d_scatter(image: np.ndarray, title: str, filename: str, labels: Optional[np.ndarray] = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    blues = image[..., 0]
    greens = image[..., 1]
    reds = image[..., 2]
    scatter = ax.scatter(reds, blues, greens, c=labels, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Red'), ax.set_ylabel('Blue'), ax.set_zlabel('Green')
    if labels is not None:
        legend = ax.legend(*scatter.legend_elements(), loc='lower left', title='Clusters')
        ax.add_artist(legend)
    plt.savefig(f'./figures/{filename}')
    plt.show()


def reconstruct_image(labels: np.ndarray, means: np.ndarray, shape: Tuple[int, int, int],
                      plot: bool = False, title: Optional[str] = None, filename: Optional[str] = None,
                      colorspace: Optional[int] = None) -> np.ndarray:
    if plot and not filename:
        raise ValueError('Must specify a filename if plotting.')

    means = means.astype(int)
    reconstructed_vector = np.array([means[label] for label in labels])
    reconstructed_image = reconstructed_vector.reshape(shape).astype(np.uint8)

    if colorspace is not None:
        reconstructed_image = cv2.cvtColor(reconstructed_image, colorspace)

    if plot:
        plt.figure()
        plt.imshow(reconstructed_image[..., ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title(f'{title}')
        plt.savefig(f'./figures/{filename}')
        plt.show()

    return reconstructed_image


if __name__ == '__main__':
    vectorized_image, original_shape = process_image('assets/HE_img.PNG', plot=False)

    for num_clusters in range(2, 8):
        gmm_labels, gmm_means = fit_gmm_dist(vectorized_image, num_clusters, 100, display=False)
        gmm_image = reconstruct_image(gmm_labels, gmm_means, original_shape,
                                      plot=False, title=f'Reconstructed image from {num_clusters} clusters',
                                      filename=f'rgb_gmm_{num_clusters}.png')

    for color_code, identifier, inverse in zip([cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2YCR_CB],
                                               ['hsv', 'lab', 'ycr_cb'],
                                               [cv2.COLOR_HSV2BGR, cv2.COLOR_LAB2BGR, cv2.COLOR_YCR_CB2BGR]):
        vectorized_image, original_shape = process_image('assets/HE_img.PNG', plot=False, colorspace=color_code)
        gmm_labels, gmm_means = fit_gmm_dist(vectorized_image, n_clusters=4, max_iterations=100, display=False)
        gmm_image = reconstruct_image(gmm_labels, gmm_means, original_shape,
                                      plot=True, title=f'Reconstructed image from {identifier} colorspace',
                                      filename=f'{identifier}_gmm.png', colorspace=inverse)
