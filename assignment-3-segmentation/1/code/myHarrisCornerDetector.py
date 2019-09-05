import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import time


def gaussian_mask(n, sigma=None):
    if sigma is None:
        sigma = 0.3*(n//2) + 0.8
    X = np.mgrid[:n]
    Y, X = np.mgrid[:n, :n]
    Y = Y - n//2
    X = X - n//2
    kernel = np.exp(-(X**2+Y**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def smoothen(I, n=3, sigma=None):
    kernel = gaussian_mask(n, sigma)
    h, w = I.shape
    n = n//2
    I_s = np.zeros(I.shape)
    for y in range(n, h-n):
        for x in range(n, w-n):
            patch = I[y-n:y+n+1, x-n:x+n+1]
            I_s[y, x] = np.sum(patch * kernel)
    return I_s


def corner_harris(I, n_g=5, n_w=5, k=0.025):
    h, w = I.shape
    sobel_x = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])
    sobel_y = np.array([[1,  2,  1],
                        [0,  0,  0],
                        [-1, -2, -1]])
    print('- Calculating Image Derivatives')
    I_x = np.zeros(I.shape)
    I_y = np.zeros(I.shape)
    D = np.zeros((h, w, 2, 2))
    for y in range(1, h-1):
        for x in range(1, w-1):
            patch = I[y-1:y+2, x-1:x+2]
            I_x[y, x] = np.sum(patch * sobel_x)
            I_y[y, x] = np.sum(patch * sobel_y)
    print('- Smoothing Derivatives')
    I_x = smoothen(I_x, n_g)
    I_y = smoothen(I_y, n_g)
    for y in range(1, h-1):
        for x in range(1, w-1):
            a, b = I_x[y, x], I_y[y, x]
            D[y, x] = np.array([[a*a, a*b],
                                [a*b, b*b]])
    kernel = gaussian_mask(n_w)
    kernel = np.dstack([kernel]*4)
    kernel = kernel.reshape(n_w, n_w, 2, 2)
    L_1 = np.zeros(I_x.shape)
    L_2 = np.zeros(I_x.shape)
    print('- Calculating Structure Tensor and Eigenvalues')
    n = n_w//2
    for y in range(n, h-n):
        for x in range(n, w-n):
            A = kernel * D[y-n:y+n+1, x-n:x+n+1]
            A = A.sum((0, 1))
            a, b, c, d = A.ravel()
            t1 = (a+d)/2
            t2 = np.sqrt((a-d)**2+4*b*c)/2
            L_1[y, x] = t1+t2
            L_2[y, x] = t1-t2
        print('Progress: {:.02f}{}'.format(
            100*(y-2*n)/(h-4*n-2), '%'), end='\r')
    print('- Calculating Corner-ness')
    C = L_1*L_2 - k*(L_1+L_2)**2
    return C, I_x, I_y, L_1, L_2


img_path = '../data/boat.jpg'
img = np.array(Image.open(img_path).convert('L'))
img = (img - img.min())/(img.max()-img.min())
start = time.time()
C, I_x, I_y, L_1, L_2 = corner_harris(img, k=0.065)
C = (C - C.min())/(C.max()-C.min())
stop = time.time()
print('Time Elasped: {:.02f}sec'.format(stop - start))


plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title('Source Image')
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.title('Sobel X derivative')
plt.imshow(I_x, cmap='gray')
plt.subplot(133)
plt.title('Sobel Y derivative')
plt.imshow(I_y, cmap='gray')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('Eigenvalue 1')
plt.imshow(L_1, cmap='jet')
plt.subplot(122)
plt.title('Eigenvalue 2')
plt.imshow(L_2, cmap='jet')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Source Image')
plt.subplot(132)
plt.imshow(C)
plt.title('Corner-ness Map')
plt.subplot(133)
plt.imshow(img/2+(C > 0.5), cmap='gray')
plt.title('Detected Corners')
plt.tight_layout()
plt.show()
