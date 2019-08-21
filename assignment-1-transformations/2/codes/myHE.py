import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_cdf(img):
    h, w = img.shape
    pdf = np.histogram(img, 256, (0, 256))[0] / (h*w)
    cdf = 255 * pdf.cumsum()
    cdf = cdf.astype(np.uint8)
    return cdf


def applyHE(img):
    cdf = get_cdf(img)
    he_img = np.zeros_like(img, np.uint8)
    he_img = cdf[img]
    return he_img


def histogramEqualize(img_path):
    img = Image.open(img_path)
    img = np.array(img)

    if img.ndim == 3:
        img_slices = [applyHE(img[:, :, i]) for i in range(3)]
        he_img = np.dstack(img_slices)
    elif img.ndim == 2:
        he_img = applyHE(img)
    else:
        raise ValueError

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(he_img, cmap='gray')
    plt.title('Histogram Equalized Image')

    plt.subplot(2, 2, 3)
    plt.hist(img.ravel(), 64, (0, 256))
    plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.hist(he_img.ravel(), 64, (0, 256))
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    histogramEqualize('../data/TEM.png')
