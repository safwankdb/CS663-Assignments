import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_tile(img, y, x, k):
    h, w = img.shape
    p1 = (max(0, y-k), max(0, x-k))
    p2 = (min(y+k, h-1), min(x+k, w-1))
    return p1, p2


def get_cdf(img, threshold=0.04):
    pdf = np.histogram(img, 256, (0, 256))[0]

    # Limiting Contrast
    extra_mass = 0
    pdf = pdf / (img.shape[0]*img.shape[1])
    for i, p in enumerate(pdf):
        if p > threshold:
            extra_mass += p - threshold
            pdf[i] = threshold
    pdf += extra_mass / len(pdf)

    cdf = 255 * pdf.cumsum()
    cdf = cdf.astype(np.uint8)
    return cdf


def applyHE(img):
    cdf = get_cdf(img)
    he_img = np.zeros_like(img, np.uint8)
    he_img = cdf[img]
    return he_img


def applyCLAHE(img, k):
    new_img = np.zeros_like(img, np.uint8)
    h, w = img.shape
    for i in range(0, h):
        for j in range(0, w):
            p1, p2 = get_tile(img, i, j, k)
            tile = img[p1[0]:p2[0], p1[1]:p2[1]]
            new_tile = applyHE(tile)
            new_img[p1[0]:p2[0], p1[1]:p2[1]] = new_tile
    return new_img


def CLAHE(img_path, window_size=50):
    """
    img_path    : Image to apply CLAHE algorithm\n
    window_size : Window size for the algorithm\n
    Returns target image as numpy array
    """
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    assert window_size <= min(h, w), "Window size is too large"
    k = window_size//2

    if img.ndim == 3:
        img_slices = [applyCLAHE(img[:, :, i], k) for i in range(3)]
        img_clahe = np.dstack(img_slices)
    elif img.ndim == 2:
        img_clahe = applyCLAHE(img, k)
    else:
        raise ValueError

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_clahe, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Contrast Limited Adaptive Histogram Equalized Image')
    plt.show()


if __name__ == "__main__":
    img_path = '../data/chestXray.png'
    CLAHE(img_path)
