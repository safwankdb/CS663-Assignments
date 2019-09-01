import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def make_filter(r):
    Y, X = np.mgrid[:2*r+1, :2*r+1]
    dist = (Y - r)**2 + (X - r)**2
    mask = dist <= r**2
    mask = mask / np.sum(mask)


def equidistant_points(x, y, h, w, dist):
    points = set()
    for i in range(dist+1):
        j = np.round((dist**2-i**2)**(0.5))
        points.add((j, i))
        points.add((i, j))
    points = np.array(list(points)).astype(np.int)
    p1 = points + [y, x]
    p2 = points * [-1, 1] + [y, x]
    p3 = points * [1, -1] + [y, x]
    p4 = points * [-1, -1] + [y, x]
    points = np.concatenate([p1, p2, p3, p4])
    final_points = []
    for p in points:
        if p[0] >= 0 and p[0] < h:
            if p[1] >= 0 and p[1] < w:
                final_points.append(list(p))
    return final_points



def min_dist(x, y, mask, anchor, thresh):
    h, w = mask.shape
    p, q = anchor
    low = 1
    high = ((x - p)**2 + (y - q)**2)**0.5
    while True:
        if low >= thresh:
            return thresh
        dist = np.floor((low+high)/2).astype(np.int)
        flag = False
        points = equidistant_points(x, y, h, w, dist)
        for p in points:
            if mask[p[0], p[1]] > 0:
                flag = True
                break
        if flag:
            high = dist
        else:
            low = dist
        if high - low < 2:
            return dist



def plot_r(img, mask, threshold):
    h, w = mask.shape
    r_map = np.zeros_like(mask).astype(np.int)
    anchor = None
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                anchor = (x, y)
                break
        if anchor is not None:
            break
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                continue
            r_map[y, x] = min_dist(x, y, mask, anchor,threshold)
    return r_map


if __name__ == "__main__":
    mask = np.array(Image.open('../images/mask1.png'))
    img = np.array(Image.open('../data/flower.jpg'))
    r_map = plot_r(img, mask, 20)

    plt.figure()
    plt.subplot(131)
    plt.imshow(mask, cmap='gray')
    plt.subplot(132)
    plt.imshow(img)
    plt.subplot(133)
    plt.imshow(r_map, cmap='jet')
    plt.show()
