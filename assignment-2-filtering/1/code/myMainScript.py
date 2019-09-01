import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from mySpatiallyVaryingKernel import min_dist, blur


def plot_r(img, mask, threshold):
    h, w = mask.shape
    r_map = np.zeros((h, w), np.int)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                r_map[y, x] = min_dist(x, y, mask, threshold)
        print('{:.02f}%'.format((y+1)*100/h), end='\r')
    print('\nDone', flush='True')
    return r_map


mask1 = np.array(Image.open('../images/mask1.png').resize((212, 141)))
img1 = np.array(Image.open('../data/flower.jpg').resize((212, 141)))

mask2 = np.array(Image.open('../images/mask2.png').resize((550, 366)))
img2 = np.array(Image.open('../data/bird.jpg').resize((550, 366)))

plt.figure()
plt.subplot(231)
plt.imshow(mask1, cmap='gray')

plt.subplot(232)
img_masked1 = np.dstack([(mask1>0)*img1[:,:,i] for i in range(3)])
plt.imshow(img_masked1)

plt.subplot(233)
img_masked1 = np.dstack([(mask1==0)*img1[:,:,i] for i in range(3)])
plt.imshow(img_masked1)

plt.subplot(234)
plt.imshow(mask2, cmap='gray')

plt.subplot(235)
img_masked2 = np.dstack([(mask2>0)*img2[:,:,i].copy() for i in range(3)])
plt.imshow(img_masked2)

plt.subplot(236)
img_masked2 = np.dstack([(mask2==0)*img2[:,:,i].copy() for i in range(3)])
plt.imshow(img_masked2)

plt.show()

r_map1 = plot_r(img1, mask1, 10)
blur(img1, mask1, r_map1)
r_map2 = plot_r(img2, mask2, 20)

plt.figure()
plt.subplot(231)
plt.imshow(mask1, cmap='gray')
plt.subplot(232)
plt.imshow(img1)
plt.subplot(233)
plt.imshow(r_map1, cmap='jet')
# plt.show()

plt.subplot(234)
plt.imshow(mask2, cmap='gray')
plt.subplot(235)
plt.imshow(img2)
plt.subplot(236)
plt.imshow(r_map2, cmap='jet')
plt.show()
