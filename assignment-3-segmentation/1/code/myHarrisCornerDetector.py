import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import time


def gaussian_mask(n, sigma=None):
    if sigma is None:
        sigma = 0.3*(n//2) + 0.8
    X = np.arange(-(n//2), n//2+1)
    kernel = np.exp(-(X**2)/(2*sigma**2))
    return kernel


def seperable_conv(I, filter_x, filter_y):
    h, w = I.shape[:2]
    n = filter_x.shape[0] // 2
    I_a = np.zeros(I.shape)
    I_b = np.zeros(I.shape)
    for x in range(n, w-n):
        patch = I[:, x-n:x+n+1]
        I_a[:, x] = np.sum(patch * filter_x, 1)
    filter_y = np.expand_dims(filter_y, 1)
    for y in range(n, h-n):
        patch = I_a[y-n:y+n+1, :]
        I_b[y, :] = np.sum(patch * filter_y, 0)
    return I_b


def corner_harris(I, n_g=5, n_w=5, k=0.06):
    h, w = I.shape
    sobel_1 = np.array([-1, 0, 1])
    sobel_2 = np.array([1, 2, 1])
    print('- Calculating Image Derivatives')
    I_x = seperable_conv(I, sobel_1, sobel_2)
    I_y = seperable_conv(I, sobel_2, sobel_1)
    print('- Smoothing Derivatives')
    g_kernel = gaussian_mask(n_g)
    I_x = seperable_conv(I_x, g_kernel, g_kernel)
    I_y = seperable_conv(I_y, g_kernel, g_kernel)
    print('- Calculating Structure Tensors')
    D_temp = np.zeros((h, w, 2, 2))
    for y in range(1, h-1):
        for x in range(1, w-1):
            a, b = I_x[y, x], I_y[y, x]
            D_temp[y, x] = np.array([[a*a, a*b],
                                     [a*b, b*b]])
    n = n_w//2
    g_filter = gaussian_mask(n_w)
    g_filter = np.dstack([g_filter]*4).reshape(n_w, 2, 2)
    D = seperable_conv(D_temp, g_filter, g_filter)
    print('- Calculating Eigenvalues')
    L_1 = np.zeros(I.shape)
    L_2 = np.zeros(I.shape)
    for y in range(n, h-n):
        for x in range(n, w-n):
            a, b, c, d = D[y, x].ravel()
            t1 = (a+d)/2
            t2 = np.sqrt((a-d)**2+4*b*c)/2
            L_1[y, x] = t1-t2
            L_2[y, x] = t1+t2
    print('- Calculating Corner-ness')
    C = L_1*L_2 - k*(L_1+L_2)**2
    return C, I_x, I_y, L_1, L_2


img_path = '../data/boat.jpg'
img = np.array(Image.open(img_path).convert('L'))
img = (img - img.min())/(img.max()-img.min())
start = time.time()
C, I_x, I_y, L_1, L_2 = corner_harris(img, k=0.06)
C = (C - C.min())/(C.max()-C.min())
stop = time.time()
print('Done!\nTime Elapsed: {:.02f}s'.format(stop - start))


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('$I_x$')
plt.imshow(I_x, cmap='gray')
plt.subplot(122)
plt.title('$I_y$')
plt.imshow(I_y, cmap='gray')
plt.tight_layout()
plt.savefig('../report/1.jpg')
plt.show()


plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.title(r'$\lambda_1$')
plt.imshow(L_1, cmap='gnuplot')
plt.colorbar()
plt.subplot(122)
plt.title(r'$\lambda_2$')
plt.imshow(L_2, cmap='gnuplot')
plt.colorbar()
plt.tight_layout()
plt.savefig('../report/2.jpg')
plt.show()


plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.imshow(C-0.457, cmap='gnuplot')
plt.title('Corner-ness Map')
plt.colorbar()
plt.subplot(122)
plt.imshow(img/2+2*C*(C >= 0.457), cmap='gnuplot')
plt.title('Detected Corners')
plt.tight_layout()
plt.savefig('../report/3.jpg')
plt.show()
