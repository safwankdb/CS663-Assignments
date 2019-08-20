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


def applyHM(src, ref):
    ref_cdf = get_cdf(ref)
    src_he = applyHE(src)
    inv_cdf = np.zeros(256, np.int) - 1
    for i, p in enumerate(ref_cdf):
        if p == -1:
            inv_cdf[p] = i
    for i, p in enumerate(inv_cdf):
        if p == -1:
            min_diff = 1000
            for j, p1 in enumerate(ref_cdf):
                if abs(i - p1) < min_diff:
                    min_diff = abs(i - p1)
                    inv_cdf[i] = j
    inv_cdf = inv_cdf.astype(np.uint8)
    hm_img = np.zeros_like(src, np.uint8)
    hm_img = inv_cdf[src_he]
    return hm_img


def histogramMatch(src_path, ref_path):
    src = Image.open(src_path)
    src = np.array(src)
    ref = Image.open(ref_path)
    ref = np.array(ref)

    assert src.ndim == ref.ndim
    if src.ndim == 3:
        img_slices = [applyHM(src[:, :, i], ref[:, :, i]) for i in range(3)]
        hm_img = np.dstack(img_slices)
        hm_img = Image.fromarray(hm_img)
    elif src.ndim == 2:
        hm_img = applyHM(src, ref)
        hm_img = Image.fromarray(hm_img)
    else:
        raise ValueError

    plt.subplot(2, 3, 1)
    plt.imshow(src, cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.title('Source Image')

    plt.subplot(2, 3, 2)
    plt.imshow(ref, cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.title('Reference Image')

    plt.subplot(2, 3, 3)
    plt.imshow(hm_img, cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.title('Histogram Matched Image')

    plt.subplot(2, 3, 4)
    plt.hist(src.ravel(), 256)
    plt.xticks([]),plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.hist(ref.ravel(), 256)
    plt.xticks([]),plt.yticks([])

    plt.subplot(2, 3, 6)
    plt.hist(np.array(hm_img).ravel(), 256)
    plt.xticks([]),plt.yticks([])
    plt.show()


if __name__ == "__main__":
    histogramMatch('../data/retina.png', '../data/church.png')
