import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def linearContrastStretch(img_path):
    img = Image.open(img_path)
    img = np.array(img)

    img_min, img_max = img.min((0, 1)), img.max((0, 1))
    curr_range = img_max - img_min
    img_streched = ((img - img_min)/curr_range)*255
    img_streched = img.astype(np.uint8)

    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    if img.ndim == 3:
        plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(img_streched, cmap='gray')
    plt.title('Contrast Stretched Image')
    if img.ndim == 3:
        plt.colorbar()
    plt.show()


if __name__ == "__main__":
    linearContrastStretch('../data/canyon.png')
