import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_tile(img, x, y, k):
    h, w = img.shape
    p1 = (max(0, x-k), max(0, y-k))
    p2 = (min(x+k, h-1), min(y+k, w-1))
    return p1, p2


def get_cdf(img):
    h, w = img.shape
    pdf = np.histogram(img, 256, (0, 256))[0] / (h*w)
    cdf = pdf.cumsum()
    cdf = 255 * (cdf - cdf.min())/(cdf.max() - cdf.min())
    cdf = cdf.astype(np.uint8)
    return cdf


def applyHE(img):
    cdf = get_cdf(img)
    he_img = np.zeros_like(img, np.uint8)
    he_img = cdf[img]
    return he_img


def applyCLAHE(img, k):
    new_img = np.zeros_like(img, np.uint8)
    for i in range(k, img.shape[0] - k):
        for j in range(k, img.shape[1] - k):
            p1, p2 = get_tile(img, i, j, k)
            tile = img[p1[0]:p2[0], p1[1]:p2[1]]
            new_img[p1[0]:p2[0], p1[1]:p2[1]] = applyHE(tile)
    return new_img


def CLAHE(img_path, window_size=25):
    """
    img_path    : Image to apply CLAHE algorithm\n
    window_size : Window size for the algorithm\n
    Returns target image as numpy array
    """
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    assert window_size % 2 == 1, "Window size must be odd"
    assert window_size <= min(h, w), "Window size is too large"
    k = window_size//2
    img_clahe = applyCLAHE(img, k)
    
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(2, 1, 2)
    plt.imshow(img_clahe, cmap='gray')
    plt.title('CLHE Image')
    plt.show()


if __name__ == "__main__":
    img_path = '../data/chestXray.png'
    CLAHE(img_path)
