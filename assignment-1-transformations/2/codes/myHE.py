import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


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

    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 1, 2)
    plt.imshow(he_img, cmap='gray')
    plt.title('Histogram Equalized Image')
    plt.show()


if __name__ == "__main__":
    histogramEqualize('../data/retina.png')
