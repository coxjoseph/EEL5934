import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.exposure import equalize_hist
import cv2
from histomicstk.preprocessing.color_deconvolution import color_deconvolution_routine


def pca(image: np.ndarray, num_components: int = 3) -> list:
    """
    Method to display and compute PCA for an inputted image. Returns a list of
    arrays defined by the principal components.
    :param image: Input RGB image
    :param num_components: number of pca components
    :return:
    """
    pixels = np.reshape(image, (-1, 3))

    pca_solver = PCA(n_components=num_components)
    pca_solver.fit(pixels)  # Perform PCA
    components = pca_solver.components_

    reconstructed_images = []
    for i in range(num_components):
        reconstructed_pixels = np.dot(pixels, components.T[i])
        reconstructed_image = np.reshape(reconstructed_pixels, carcinoma.shape[:-1])
        reconstructed_image_clipped = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        reconstructed_images.append(reconstructed_image_clipped)

    fig, axes = plt.subplots(1, num_components, figsize=(15, 5))
    for i in range(num_components):
        axes[i].imshow(reconstructed_images[i], cmap='gray')
        axes[i].set_title(f'Principal Component {i + 1}')
        axes[i].axis('off')

    plt.show()
    return reconstructed_images


def show_rgb_channels(image: np.ndarray, cmap='gray') -> None:
    """
    Plot individual channels of an RGB image using the provided colormapping
    :param image: Input image
    :param cmap: color mapping (must be valid pyplot cmap)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        axes[i].imshow(image[:, :, i], cmap=cmap)
        axes[i].set_title(f'Color {colors[i]}')
        axes[i].axis('off')

    plt.show()


def split_hed_channels(image: np.ndarray) -> np.ndarray:
    """
    Split image into Haematoxylin-Eosin-DAB (HED) colorspace, show the H and E channels, and return converted image
    :param image: original RBG image
    :return: Image converted to HED colorspace
    """

    hed = rgb2hed(image)

    colors = ['Hematoxylin ', 'Eosin']
    fig, axes = plt.subplots(1, len(colors), figsize=(15, 5))
    for i in range(len(colors)):
        axes[i].imshow(hed[:, :, i], cmap='gray')
        axes[i].set_title(f'Stain {colors[i]}')
        axes[i].axis('off')

    plt.show()
    return hed


def binarize(image: np.ndarray, flip_colors: bool = False) -> np.ndarray:
    threshold_value = threshold_otsu(image)
    if flip_colors:
        binary_image = (image <= threshold_value).astype(np.uint8) * 255
    else:
        binary_image = (image > threshold_value).astype(np.uint8) * 255

    return binary_image


def compare_to_ground_truth(binarized_image: np.ndarray, ground_truth: np.ndarray) -> float:
    new_shape = (min(binarized_image.shape[0], ground_truth.shape[0]),
                 min(binarized_image.shape[1], ground_truth.shape[1]))

    binarized_image = cv2.resize(binarized_image, new_shape[::-1])
    ground_truth = cv2.resize(ground_truth, new_shape[::-1])

    same_pixels = np.sum(binarized_image == ground_truth)
    total_pixels = np.prod(binarized_image.shape)

    percentage_same = (same_pixels / total_pixels) * 100
    return percentage_same


def rand_mat_hist(image_shape: tuple, n_levels: int = 500):
    og_img = np.zeros(image_shape)

    # The original file claims to be iterating through columns but it is rows
    for i, row in enumerate(og_img):
        indices = np.random.randint(1, n_levels, size=(image_shape[1]))

        for j in range(len(indices)):
            og_img[i, j] = 1 / indices[j]

    hist, bins = np.histogram(og_img.flatten(), bins=256)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].imshow(og_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].bar(bins[:-1], hist, width=1/256)
    axes[0, 1].set_title('Histogram of Image')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')

    equalized_image = equalize_hist(og_img)
    axes[1, 0].imshow(equalized_image)
    axes[1, 0].set_title('Equalized Image')
    axes[1, 0].axis('off')

    hist, bins = np.histogram(equalized_image.flatten(), bins=1000)
    axes[1, 1].bar(bins[:-1], hist, width=1/256)
    axes[1, 1].set_title('Histogram of Equalized Image')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')

    plt.show()
    return equalized_image


if __name__ == '__main__':
    # Question 1:
    carcinoma_image = Image.open('assets/Carcinoma_In_Situ.jpg')
    carcinoma = np.array(carcinoma_image)

    # Q1a
    pca_reconstructed = pca(image=carcinoma)

    # Q1b
    show_rgb_channels(carcinoma)

    # Q1c
    hed_deconv = split_hed_channels(carcinoma)

    # Q1d
    best_pca = pca_reconstructed[1]
    best_rgb = carcinoma[:, :, 0]
    best_hed = hed_deconv[:, :, 0]

    bests = [best_pca, best_rgb, best_hed]
    labels = ['PCA', 'RGB', 'HED']
    flips = [False, True, False]

    gt = Image.open('assets/GT.png')
    gt = np.array(gt).astype(np.uint8) * 255
    for best, label, flip in zip(bests, labels, flips):
        binarized_best = binarize(best, flip_colors=flip)
        percent_common = compare_to_ground_truth(binarized_best, gt)
        print(f'{label} had {round(percent_common, 2)}% of pixels in common with the ground truth.')

    # Q2
    _ = rand_mat_hist((200, 200), n_levels=500)
    _ = rand_mat_hist((200, 200), n_levels=1000)
    _ = rand_mat_hist((200, 200), n_levels=200)

    eyeball_image = Image.open('assets/DR_eyeball.jpg')
    bw_eyeball = eyeball_image.convert('L')
    bw_eyeball = np.array(bw_eyeball)

    plt.figure()
    plt.imshow(bw_eyeball, cmap='gray')
    plt.title('Black and white eye image')
    plt.axis('off')
    plt.show()
    eq_image = equalize_hist(bw_eyeball)
    plt.figure(figsize=(15, 5))
    plt.imshow(eq_image, cmap='gray')
    plt.title('Histogram Equalized Image')
    plt.axis('off')
    plt.show()

    clahe_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    plt.figure(figsize=(15, 5))
    plt.imshow(eq_image, cmap='gray')
    plt.title('CLAHE Enhanced Image')
    plt.axis('off')
    plt.show()

    # Q3
    # Q3b
    # stain vectors from QuPath
    vectors = np.array([[0.106, 0.821, 0.561],
               [0.289, 0.573, 0.767],
               [0.845, 0.223, -0.485]])

    wsi_patch = Image.open('assets/HW4_Example_WSI_Patch.PNG')
    wsi_patch = np.array(wsi_patch)

    deconvolved_images, _, _ = color_deconvolution_routine(wsi_patch, vectors)
    cd_34, cd_45, resid = deconvolved_images[:, :, 0], deconvolved_images[:, :, 1], deconvolved_images[:, :, 2]
    fig, axes = plt.subplots(3, 1, figsize=(15, 5))
    axes[0].axis('off')
    axes[0].set_title('CD 34 Deconvolution')
    axes[0].imshow(cd_34, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('CD 45 Deconvolution')
    axes[1].imshow(cd_45, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Residual Deconvolution')
    axes[2].imshow(resid, cmap='gray')
    plt.show()

    cd_34_binarized = binarize(cd_34)
    cd_45_binarized = binarize(cd_45)
    resid_binarized = binarize(resid)

    stained_area = np.sum(np.bitwise_or(cd_34_binarized, cd_45_binarized) / 255)
    percentage_total = (stained_area / np.size(cd_34_binarized)) * 100
    print(f'Stains occupy {round(percentage_total, 2)}% of the total image')

